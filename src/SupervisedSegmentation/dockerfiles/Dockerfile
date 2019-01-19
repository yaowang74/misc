# Python version
FROM python:3.6.6

# Working directory
WORKDIR /home/h211803/Documents/BitBucketReository/smart_energy/src/SupervisedSegmentation

# Copy requirements.txt
COPY requirements.txt /tmp

# Install any needed packages specified in requirements.txt
RUN pip install -r /tmp/requirements.txt 

# Install graphviz
RUN pip install graphviz

#
RUN apt-get update && apt-get install -y graphviz

# Copy application config file
COPY ./supervisedsegmentation/app_config.ini /tmp

# Copy data set config file
COPY ./supervisedsegmentation/regression_tree_sample_config.csv /tmp

# Copy common module codes into the container
COPY ./supervisedsegmentation/CommonModules /tmp/CommonModules

# Copy main codes into container
COPY ./supervisedsegmentation/supervisedsegmentation.py /tmp

# Run application
ENTRYPOINT ["python", "/tmp/supervisedsegmentation.py"]


