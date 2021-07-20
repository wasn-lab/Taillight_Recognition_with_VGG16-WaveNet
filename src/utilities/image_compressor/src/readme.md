### image_compressor_node

Example:

```
image_compressor_node -input_topic /cam/front_bottom_60/raw -output_topic /cam/front_bottom_60/jpg -quality 85
```

Compression ratio:
```
rostopic bw /cam/front_bottom_60/raw

average: 14.56MB/s
	mean: 0.62MB min: 0.62MB max: 0.62MB window: 21
average: 14.11MB/s
	mean: 0.62MB min: 0.62MB max: 0.62MB window: 43
average: 13.97MB/s
	mean: 0.62MB min: 0.62MB max: 0.62MB window: 65
average: 13.91MB/s
	mean: 0.62MB min: 0.62MB max: 0.62MB window: 87
```


|quality|jpg bw (MB/s)|
|-------|-------------|
|95|1.74|
|90|1.60|
|85|1.37|
|80|1.27|
|75|1.22|
|70|1.09|
|65|0.74|
|60|0.71|
|50|0.66|
|40|0.51|
|30|0.41|
|20|0.3|
|10|0.16|

