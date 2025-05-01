import firebase_admin
from firebase_admin import credentials, storage
import os

# Initialize Firebase
cred = 
credentials.Certificate("/Users/mac/cardio-echo/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'cardioecho-9af8c.appspot.com'
})

# Configure emulator
os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:9201"

# Upload video
blob = bucket.blob('videos/sample.mp4')
blob.upload_from_filename("~/cardio-echo/sample.mp4")
print("Uploaded sample.mp4")
