import sys
sys.path.append("../utoronto_demo") 
from north import NorthC9
from Locator import *
import os
from prefect import flow,serve
from prefect_aws.s3 import S3Bucket

s3_bucket_block = S3Bucket.load("awss3")

@flow(log_prints=True,persist_result=True,result_storage=s3_bucket_block)
def move_track(loc):
    c9 = NorthC9('A', network_serial='AU06CNCF')
    try:
        c9.move_axis(7, loc)
    except:
        print("Error moving track")

if __name__ == "__main__":
    flow1 = move_track.to_deployment(name="move_track")

    serve(flow1)

