# run for THE FIRST TIME the docker file with a run.sh script below
#!/bin/sh
if [ -z "$IMAGETAG" ]; then
    IMAGETAG="latest"
fi

# IMPORTANT: set the correct volume when running this
docker run --rm -it -v `dirname ${PWD}`:/home/${USER}/work \
  mycontainer:${IMAGETAG}