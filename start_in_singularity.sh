if [ "a" = "a" ]; then
    echo "> Booting into bash inside Singularity container..."
    export PROMPT_COMMAND="echo -n \[\ Singularity \]\ "
    exec bash
fi