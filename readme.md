
A series of classical anomaly detection algorithms tailored for BAG files used in NOAA OCS.

## Installation
```
$ conda env create -f environment.yml
$ conda activate anomaly
```

## Usage
Dashboard
```
$ python anomaly.py --dashboard
```

CLI
```
$ python anomaly.py E01001_MB_VR_MLLW_2of5.bag --method lof --output results.tiff
```