# lane2array
This lane2array.py helps generate numpy arrays for IPP model from ITRI labeled Lanelet2 maps.

## Environment Setup
### 1. Clone the official Lanelet2 repository.
```
git clone https://github.com/fzi-forschungszentrum-informatik/Lanelet2.git
```

### 2. Go to the root directory. Replace the existing dockerfile with ours, then build it.

Please do replace, we have added some requiring packages.
```
docker build -t lanelet2 .
```

### 3. Put our code into your cloned lanelet2/lanelet2_maps directory.
> lane2array.py

### 4. Put your Lanelet2 file into lanelet2/lanelet2_maps/res directory.
> ITRI_lanelet2_map.osm
>
If you cloned the whole itriadv repo, you should be able to find maps in `/src/sensing/map/map_loader/data/`.
Get your map file here [ITRI map repo](https://gitlab.itriadv.co/self_driving_bus/itriadv/tree/master/src/sensing/map/map_loader/data)

### 5. Run the bottom command in order to mount the code into the container.
The running container would be named as "mylanelet2". 
You should be in your cloned lanelet2 root directory (instead of lanelet2/lanelet2_maps/) when you run the following command, since we set the mounting source path here as "$(pwd)".
```
docker run -d \
  -it \
  --name mylanelet2 \
  --mount type=bind,source="$(pwd)",target=/home/developer/workspace/src/lanelet2 \
  lanelet2_src:latest
```

### 6. After building and running this docker for the first time, you can from now on execute the docker like below.

```
docker exec -it mylanelet2 /bin/bash
```

## Running the Code

### 1. Get into the running container.
Start the container and execute it.
```
docker start mylanelet2
docker exec -it mylanelet2 /bin/bash
```

### 2. Go to where lane2array.py is.
```
cd src/lanelet2/lanelet2_maps
```


### 3. Run the code with arguments.
| Argument | Definition | Default  |
| ------------- | ---------- |----------|
| map      | the name of your map file | ITRI_lanelet2_map |
| output   | 0: output only numpy array file, 1: output numpy array file and figure of the map | 0
| type   | 0: vehicle, 1: motorcycle, 2: pedestrian| 0

Example:
1. Simply test if the code can run, generating the ITRI_ADV map mask.
`python lane2array.py`
The terminal should look like this.
![](https://i.imgur.com/vRe3ec8.png)

And now you should be able to find the `ITRI_lanelet2_map.npy` file in your local lanelet2/lanelet2_maps folder.


2. Test it again with the default file but generate both npy array mask and figure this time.
`python lane2array.py --output 1`
Now you can see a `ITRI_lanelet2_map_fig.png` looking like this in the local folder.
![](https://i.imgur.com/E30wIbo.png)


3. Run the code with your map. You don't need to type in ".osm".
`python lane2array.py --map your_map_name`


4. Run the code with target pedestrian and output the figure.
`python lane2array.py --output 1 --target 2`
![](https://i.imgur.com/GGYrsVx.png)


