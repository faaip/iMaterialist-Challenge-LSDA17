#!/bin/bash

for dir in /Users/frederikjuutilainen/Programming/KUA/iMaterialist-Challenge-LSDA17-/data/train/*
do
	ls -t $dir | tail -n +4
done
