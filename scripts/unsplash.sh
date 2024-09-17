#!/bin/bash

# Set the reference date and time 
REFERENCE_DATETIME="2024-05-09 20:00"

# Get the current date and time
CURRENT_DATETIME=$(date +%Y-%m-%d\ %H:%M)

# Convert date and time to seconds since epoch
reference_seconds=$(date -d "$REFERENCE_DATETIME" +%s)
current_seconds=$(date -d "$CURRENT_DATETIME" +%s)

# Calculate the difference in seconds
elapsed_seconds=$(( current_seconds - reference_seconds ))

# Convert elapsed seconds to hours (integer division)
elapsed_hours=$(( elapsed_seconds / 3600 ))

# Run your command with the incremented hour as an argument
python3 /Users/annamiraotoole/Documents/GitHub/sillystill/src/unplash.py $elapsed_hours >> /Users/annamiraotoole/Documents/GitHub/sillystill/scripts/cron_unsplash.log

# run the script with this command
# will run this script every hour on the hour for the 9th, 10th, and 11th of May
# 0 */1 9-11 5 * sh unsplash.sh &> cron_unsplash.log
