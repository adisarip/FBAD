#!/bin/bash
val=1024
for i in {1..10}
do 
val=`expr $val \* 2`
echo tanmay "$val"
done 