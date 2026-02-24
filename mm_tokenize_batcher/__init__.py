import numpy as np
from typing import List, Dict, Tuple, Union

__all__ = ["Batcher", "SimpleBatcher", "SortedBatcher", "ClusteredBatcher"]


class Batcher:
    """
    Base class to group images and text into batches.

    Description:
        Given a sequence (buffer) of images and optionally text
        (or eventually arbitrary modalities), a batcher will group
        the data into "logical" batches of size `batch_size` and return
        the resulting batches in a list.
        Logical means the batch itself does not modify the original data
        in any way, but metadata is computed to ensure proper preprocessing
        downstream before passes into respective tokenizers.
        Specifically, a batch at a logical level is a dictionary of structure
        similar to the following
        ```
        {
          "images": List[PILImage], # batch_size length list of images
          "text": List[str], # batch_size length list of text samples
          "resize_size": Tuple[int, int], # size the images in the batch must be resized to
          ... # any other metadata
        }
        ```
        Please refer to the respective subclasses for details
    """
    def __init__(
        self,
        batch_size: int,
        resize_size: Union[int, Tuple[int, int], str] = 'avg',
        **kwargs
    ):
        """
        Args:
            batch_size (int): The batch size
            resize_size (int, Tuple[int, int], str): The size to which to
                    resize images. Can be int (assumed square); tuple of ints
                    (the H,W to resize to); or a string either "min", "max", or "avg"
                    which computes the size as the minimum, maximum, or average of the
                    image sizes in the batch. Default is "avg"
        """
        self.batch_size = batch_size
        self.resize_size = resize_size

    def compute_resize_size(self, images: List) -> Tuple[int, int]:
        if isinstance(self.resize_size, (tuple, list)):
            return tuple(self.resize_size)
        elif isinstance(self.resize_size, int):
            return self.resize_size, self.resize_size
        elif self.resize_size == "min":
            w, h = np.array([img.size for img in images]).min(axis=0)
            return round(h), round(w)
        elif self.resize_size == "max":
            w, h = np.array([img.size for img in images]).max(axis=0)
            return round(h), round(w)
        elif self.resize_size == "avg":
            w, h = np.array([img.size for img in images]).mean(axis=0)
            return round(h), round(w)
        else:
            raise ValueError(f"Unsupported resize_size: {self.resize_size} of type {type(self.resize_size)}")

    def __call__(self, buffer: List[Dict]) -> List[Dict]:
        """
        Must be implemented by subclasses

        Args:
            buffer (List[Dict]): A list of samples to group into batches.
                    A sample is a dict of schema `{"image": <PILImage>, "text": <Optional[str]>}`
        Returns:
            A list of batches
        """
        raise NotImplementedError

class SimpleBatcher(Batcher):
    """
    Simply groups together contiguous samples of size `batch_size`
    """
    def __call__(self, buffer: List[Dict]) -> List[Dict]:
        batches = []
        curr_batch = {"images": [], "text": []}

        for sample in buffer:
            curr_batch["images"].append(sample["image"])
            # if hasattr(sample, "text"):
            if "text" in sample:
                curr_batch["text"].append(sample["text"])
            if len(curr_batch["images"]) == self.batch_size:
                batch = {
                    "images": curr_batch["images"],
                    "text": curr_batch["text"],
                    "resize_size": self.compute_resize_size(curr_batch["images"])
                }
                batches.append(batch)
                curr_batch = {"images": [], "text": []}

        # final flush
        if curr_batch["images"]:
            batch = {
                "images": curr_batch["images"],
                "text": curr_batch["text"],
                "resize_size": self.compute_resize_size(curr_batch["images"])
            }
            batches.append(batch)

        return batches

class SortedBatcher(Batcher):
    def __init__(
        self,
        batch_size: int,
        resize_size: str = 'avg',
        key_fn: Union[str, callable] = 'aspect_ratio'
    ):
        """
        Args:
            batch_size (int): The batch size
            resize_size (str): The size to which to resize images.
                    Since grouping is already done using a sorting key,
                    it is recommended to only resize using either "min", "max", or "avg"
                    shape within a batch and not resizing to a fixed size.
            key_fn (str, callable): The key function to use to sort images before batching.
                    Can be a string ('aspect_ratio' or 'area') or a callable that takes in a tuple
                    (index, width, height) and returns a sorting key over width and height. The index
                    is used to keep track of the original image after sorting.
        """
        super().__init__(batch_size, resize_size)
        if not isinstance(self.resize_size, str):
            print("resize_size is recommended to be a string. Switching to 'avg'")
            self.resize_size = "avg"

        if key_fn == 'aspect_ratio':
            self.key_fn = lambda x: x[1] / x[2] # width / height
        elif key_fn == 'area':
            self.key_fn = lambda x: x[1] * x[2] # width * height
        elif callable(key_fn):
            self.key_fn = key_fn
        else:
            raise ValueError(f"Unsupported sorting key: {key_fn}")

    def __call__(self, buffer: List[Dict]) -> List[Dict]:
        batches = []
        curr_batch = {"images": [], "text": []}

        images = [sample["image"] for sample in buffer]
        text = None
        if "text" in buffer[0]:
            text = [sample["text"] for sample in buffer]

        # Sort by key
        ar_w_idx = [(i, img.width, img.height) for i, img in enumerate(images)]
        idx_sortedby_ar = sorted(ar_w_idx, key=self.key_fn)

        for idx, _, _ in idx_sortedby_ar:
            curr_batch["images"].append(images[idx])
            if text is not None:
                curr_batch["text"].append(text[idx])

            if len(curr_batch["images"]) == self.batch_size:
                batch = {
                    "images": curr_batch["images"],
                    "text": curr_batch["text"],
                    "resize_size": self.compute_resize_size(curr_batch["images"])
                }
                batches.append(batch)
                curr_batch = {"images": [], "text": []}

        if curr_batch["images"]:
            batch = {
                "images": curr_batch["images"],
                "text": curr_batch["text"],
                "resize_size": self.compute_resize_size(curr_batch["images"])
            }
            batches.append(batch)

        return batches

class ClusteredBatcher(Batcher):
    """
    Groups samples into batches based on k-means clustering
    of aspect ratio and log area where k is buffer_size / batch_size
    """
    def __init__(
        self,
        batch_size: int,
        resize_size: str = 'avg',
        device: str = None
    ):
        """
        Args:
            batch_size (int): The batch size
            resize_size (str): The size to which to resize images.
                    Since grouping is already done via clustering,
                    it is recommended to only resize using either "min", "max", or "avg"
                    shape within a batch and not resizing to a fixed size.
            device (str, optional): Device to run kmeans on
        """
        super().__init__(batch_size, resize_size)
        if not isinstance(self.resize_size, str):
            print("resize_size is recommended to be a string. Switching to 'avg'")
            self.resize_size = "avg"

        # NOTE: due to low-data regime, cpu seems to be slightly faster
        # If it makes sense to use GPU, uncomment below and inside __call__
        # self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, buffer: List[Dict]) -> List[Dict]:
        from fastkmeans.kmeans import FastKMeans

        # Init kmeans
        kmeans = FastKMeans(
            d = 2,
            k = max(1, len(buffer) // self.batch_size),
            max_points_per_centroid = self.batch_size,
            gpu=False,
            # device = self.device,
            # use_triton = False,   # Causes untraceable CUDA errors
        )

        images = [sample["image"] for sample in buffer]
        text = None
        if "text" in buffer[0]:
            text = [sample["text"] for sample in buffer]

        # Run kmeans
        data = np.array([
            [img.width / img.height, np.log(img.width * img.height)]
            for img in images
        ], dtype=np.float32)
        min_ = data.min(axis=0)
        max_ = data.max(axis=0)
        data = (data - min_) / (max_ - min_)

        labels = kmeans.fit_predict(data)

        batches = {i: {"images": [], "text": []} for i in range(kmeans.k)}
        for i, label in enumerate(labels):
            batches[label]["images"].append(images[i])
            if text is not None:
                batches[label]["text"].append(text[i])

        output_batches = []
        for batch in batches.values():
            if not batch["images"]:
                continue

            if len(batch["images"]) <= self.batch_size:
                output_batches.append({
                    "images": batch["images"],
                    "text": batch["text"],
                    "resize_size": self.compute_resize_size(batch["images"]),
                })
            else:
                # hard split to avoid OOM
                for i in range(0, len(batch["images"]), self.batch_size):
                    imgs = batch["images"][i:i + self.batch_size]
                    txts = batch["text"][i:i + self.batch_size] if text is not None else []

                    output_batches.append({
                        "images": imgs,
                        "text": txts,
                        "resize_size": self.compute_resize_size(imgs),
                    })

        return output_batches
