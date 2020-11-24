#sudo docker run -d --name milvus_cpu_0_11_0 --device /dev/cambricon_dev0 \
#-p 19531:19531 \
#-p 19120:19120 \
#-v /home/cambricon/ylj:/workspace \
#--privileged \
#milvusdb/milvus-cpu-build-env:v0.7.0-ubuntu18.04


sudo docker start milvus_cpu_0_11_0
sudo docker exec -ti milvus_cpu_0_11_0 /bin/bash
