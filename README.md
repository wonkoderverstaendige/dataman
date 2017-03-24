# Data Manager

A collections of python tools to organize and document my data.

## Visualization
Take a peak at the data

`dm vis ~/data/2014-10-30_15-04-50`

![vis screenshot](/resources/vis_64_channels.png?raw=true)

## Preparation for klusta

### Convert
Convert .continuous files into flat binary .dat file, reordering channels according to probe file.

`dm conv ~/data/2014-10-30_15-04-50 -l ~/data/m0001_16.prb -o ~/data/testing -D 10`

### Re-reference
Create average of good channels and subtract from all channels, overwriting the unreferenced data.

`dm ref "2014-10-30_15-04-50--cg(00).dat" --inplace -l "2014-10-30_15-04-50--cg(00).prb"`

### Split
Split the averaged file into channel groups as per the (reordered) probe file and delete the combined dat file.

`"2014-10-30_15-04-50--cg(00).dat" -l "2014-10-30_15-04-50--cg(00).prb" --clean`

