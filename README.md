# multimodal-batching

Smart batching strategies for variable-resolution images in multimodal pipelines.

## Problem

Multimodal models process images at varying native resolutions, but batched inference requires uniform tensor shapes within each batch. Naive batching groups images arbitrarily and resizes them to a common size, wasting compute on padding or distortion — especially when a batch mixes tall, wide, and square images.

## Strategies

All batchers operate on a **buffer** of samples (each containing a PIL image and optional text) and return logical batches with a computed `resize_size`:

### Simple Batching
A simple albeit useful baseline method is to simply load `batch_size` images into memory, resize them to a common resolution, and feed them through the model. This is implemented in the `mm_batching.SimpleBatcher` class. The class supports resizing to a fixed size, or dynamically calculating the resize size using the minimum, average, or maximum height and width (individually) across the batch. For datasets where image size and aspect ratio vary considerably, this method will not provide the best quality after resizing compared to the following methods. However, its speed makes it the ideal implementation for datasets of fixed or roughly similar image resolutions.

### Sorted Batching
Sorted batching can be seen as the next logical step in preserving image quality after resizing. It works by taking a sorting key over image height and width, and sorting images based on this key. Batches are then formed by taking contiguous, non-overlapping subsets of the sorted images. Examples of sorting keys are aspect ratio (`img.width / img.height`), image area (`img.width * img.height`), and lexicographic sorting by e.g., first aspect ratio and then area. The implementation ensures that sorting is only done on metadata and not images themselves to reduce memory footprint.

Still, sorted batching has notable limitations. If we consider a buffer of images with a diverse range of aspect ratios and areas, it is hard to find the proper sorting key. For instance, sorting by aspect ratio might mitigate scenarios where tall and wide images are batched together, but it can cause very small and very large images to be grouped together such as a `224x224` and a `1024x1024`. Lexicographic sorting can cause "boundary effects" where strict batch sizes can fracture more logical groupings. As an example, in the following list of (aspect ratio, area) tuples: `[(1, 224**2), (1, 384**2), (1, 448**2), (1, 512**2), (1.1, ~224**2), (1.1, ~384**2), (1.1, ~448**2), (1.1, ~512**2)]`. A batch size of 4 causes `(1, 224**2)` and `(1.1, ~224**2)` to be in separate batches despite them intuitively being closer in overall resolution than say `(1, 224**2)` and `(1, 512**2)`.

The full implementation can be found under `mm_batching.SortedBatcher`.

### Clustered Batching
To properly split a buffer into batches, one needs to consider both the image area and aspect ratio. As we saw with sorted batching, collapsing these two metrics into one dimension hurts performance. An example of these tradeoffs sorting based on aspect ratio can be found in [this figure](#quality-metrics). Thus, the clustered batching approach represents each image as a 2-dimensional vector of aspect ratio and log area. It computes area in log space to incorporate scale invariance as a scale up from 2 to 4 should be treated equally to a scale up from 200 to 400, even though the latter has larger absolute difference. It then use k-means on the normalised aspect ratio and normalized log area to group the datapoints using Euclidean distance. The implementation uses [fast-kmeans](https://github.com/AnswerDotAI/fastkmeans) with number of centroids equal to the number of desired batches (typically `buffer_size / batch_size`) and each cluster size limited to `batch_size` samples.

The full implementation can be found under `mm_batching.ClusteredBatcher`.

## Usage
I developed the batchers originally for an image-text paired dataset, but they theoretically work for image-only pipelines and can be extended to include many modaltiies paired with image. I left resizing out of the batcher (1) for separation of responsibility, and (2) because resizing when computing batches is less flexible for use in downstream pipelines. Unfortunately, the method does not fit nicely into a collate function since it requires `>> batch_size` images to produce high-quality batches. It is a quality vs. speed tradeoff that works well if you have the memory and time to process buffers of images at a time, but is inappropriate for ultra low-latency use cases. Finding a good `buffer_size` is unfortunately another hyperparameter tuning task based on the needs of your pipeline, but hopefully the benchmark notebook offers you some insight on how to assess this :).

Example usage for image model
```python
from mm_tokenize_batcher import SimpleBatcher, SortedBatcher, ClusteredBatcher

# Each sample is a dict with an "image" key (PIL Image) and optional "text"
buffer = [{"image": img} for img in images]

batcher = ClusteredBatcher(batch_size=32)
batches = batcher(buffer)

for batch in batches:
    imgs = batch["images"]        # List[PIL.Image]
    h, w = batch["resize_size"]   # Target size for this batch
    # resize and tokenize ...
```

### Performance
To study the tradeoff between batching quality and batching overhead for the three batching strategies, I employ two complementary quality loss metrics: p95 resize waste, and intra-batch aspect-ratio (AR) variance. The former is calculated by taking the ratio of resized images to the original images and taking the 95th percentile. This captures worst-case padding / scaling inefficiency and correlates with peak compute and memory risk across a shard. I compare these values to the time spent per image across increasing buffer sizes.

To benchmark, generate 5000 random images of varying height and width drawn i.i.d. uniform from a range of [224, 1024]. I use a batch size of 32 and scale the buffer size by powers of 2 (i.e. `32 * 2 ** k`) up to the total number of samples in the shard. I stick with a resize strategy of "average" for each batch. Simple batching does not require a buffer, and is simply run once through the whole shard using a batch size of 32.

### Time Metrics
![Plot of batching strategy time per sample vs buffer size](assets/batching_cost.png)

When comparing the different batching strategies, one can observe that time-per-sample is negligibly impacted when using sorted batching by increasing the buffer size compared to simple batching. It does perform worse for clustered batching, although interestingly the overhead is minimized not for the smallest buffer size, but for around 2000 samples.

### Quality Metrics
![Plot of batching strategy quality metrics vs buffer size](assets/batch_quality.png)

In terms of quality with respect to the two metrics, clustered batching is the clear winner. Using either area or aspect ratio as the sorting key drastically improves p95 and variance respectively with higher batch sizes at the expense of the other. Clustered batching is the only method which reduces both.

If you'd like to analyze the batching strategies yourself or test your own solutions, refer to `benchmark_batching.ipynb`.

## Dependencies

- `numpy`
- `torch`
- `Pillow`
- `fastkmeans`
