#!/bin/bash
cd ../../../build

make ped_models_rf
make ped_models_mpi
make ped_models_mpi_txt

cd ..

