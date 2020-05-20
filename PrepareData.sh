#!/bin/bash

list_train="./data/list_ModelNet10_train.txt"
list_test="./data/list_ModelNet10_test.txt"

# 1. move all the OFF files of ModelNet10 into a directory
tmpdir="tmp"
mkdir -p $tmpdir
find ModelNet10 -type f -name "*.off" | while read i;
do
  mv $i $tmpdir
done

# 2. convert OFF files to intermediate files
pickle_train="$tmpdir/train.pickle"
pickle_test="$tmpdir/test.pickle"
command="python -u PickleOff.py $tmpdir $list_train $pickle_train"
echo $command
$command

command="python -u PickleOff.py $tmpdir $list_test $pickle_test"
echo $command
$command

# 3. convert the intermediate files to "shaperep" files that contain three shape representations (voxels, point set, and multiview images)
# Note: it is recommended that the code "PickleShapeRep.py" is executed on a terminal launched on a window manager since it uses OpenGL for multiview rendering of 3D shapes.
# Running the code via SSH connection without X forwarding would abort with an error "Exception: cannot open display".
shaperep_train="data/train.shaperep"
shaperep_test="data/test.shaperep"

command="python -u PickleShapeRep.py $pickle_train $shaperep_train"
echo $command
$command

command="python -u PickleShapeRep.py $pickle_test $shaperep_test"
echo $command
$command

# 4. delete temporary files
rm -rf $tmpdir

exit
