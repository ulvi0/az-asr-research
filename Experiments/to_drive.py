from __future__ import print_function
import os
import pickle
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Define the scopes. If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Authenticate and return the Drive service."""
    creds = None
    # Token file stores the user's access and refresh tokens.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            auth_url, _ = flow.authorization_url(prompt='consent')
            print('Please visit this URL to authorize the application:', auth_url)
            #creds = flow.run_console()
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path, folder_id=None):
    """Uploads a single file to Google Drive using a resumable upload."""
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    media = MediaFileUpload(file_path, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploading {file_name}: {int(status.progress() * 100)}% complete.")
    print(f"Upload Complete for {file_name}. File ID: {response.get('id')}")

def upload_folder(service, folder_path, parent_folder_id=None):
    """Walk through a folder and upload all files. 
    You can extend this to create corresponding folders on Drive if needed."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Uploading: {file_path}")
            upload_file(service, file_path, parent_folder_id)

def main():
    service = authenticate()
    folder_to_upload = '/whisper-finetuned'  # Replace with your folder path
    # Optionally, if you want to upload files into a specific Drive folder, specify its folder ID here.
    upload_folder(service, folder_to_upload)

if __name__ == '__main__':
    main()
