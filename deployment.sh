docker rm -f lab || echo "Not found"
docker build . -t lab
docker run --rm -d -p 80:8888 --name lab -v /opt/IDS-CNN/lab:/home/jovyan/work:rw -v /data/input:/input:rw lab