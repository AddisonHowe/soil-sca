#!/usr/bin/env bash

name=$1

resdir=results/$name
if [[ -d "$resdir" ]]; then
    read -p "Results directory $resdir exists! 
    Do you want to continue? (y/n): " answer
    case "$answer" in
        [Yy]* ) echo "Continuing...";;
        [Nn]* ) echo "Aborting."; exit 1;;
        * ) echo "Please answer [y]es or [n]o."; exit 1;;
    esac
else
    mkdir -p $resdir
fi

cp -r out/K00370/${name}/sca_results ${resdir}/
cp -r out/K00370/${name}/pymol_images ${resdir}/
cp -r out/K00370/${name}/images ${resdir}/


# cp -r out/S1Ahalabi/${name}/sca_results ${resdir}/
# cp -r out/S1Ahalabi/${name}/pymol_images ${resdir}/
# cp -r out/S1Ahalabi/${name}/images ${resdir}/
