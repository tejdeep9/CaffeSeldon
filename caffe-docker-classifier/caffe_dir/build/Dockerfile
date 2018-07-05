FROM caffe2ai/caffe2

LABEL "docker_repo"="tejdeep9"
LABEL "docker_image_name"="caffeclassifier"
LABEL "docker_image_version"="0.1"
LABEL "service_type"="MODEL"
LABEL "base_image"="caffe2ai/caffe2"
LABEL "model_name"="CaffeClassifier"
LABEL "persistence"="0"
LABEL "api_type"="REST"


COPY /requirements.txt /tmp/
COPY /seldon_requirements.txt /tmp/
RUN cd /tmp && \
    pip install --no-cache-dir -r seldon_requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir microservice
COPY ./ /microservice/
WORKDIR /microservice

EXPOSE 5000
CMD ["python","-u","microservice.py","CaffeClassifier","REST","--service-type","MODEL","--persistence","0"]