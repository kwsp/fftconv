#!/bin/bash
PROFILE_FILE=profile_$(date +%F-%H%M%S).pb.gz
CPUPROFILE=$PROFILE_FILE ./build/test_fftconv
