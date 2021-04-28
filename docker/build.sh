# build the docker filer with a build.sh script below
#!/bin/sh
if [ -z "$IMAGETAG" ]; then
    IMAGETAG="latest"
fi
docker build --force-rm -t mycontainer:${IMAGETAG} .