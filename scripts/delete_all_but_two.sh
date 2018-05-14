#!/bin/bash

for dir in /Users/frederikjuutilainen/Programming/KUA/iMaterialist-Challenge-LSDA17-/data/train/*
do
	cd $dir
	ls -t | tail -n +4 | xargs rm --
	cd ..
done
