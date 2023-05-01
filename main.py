import os
import base64
import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import email
import re
import random
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pinecone

# Authenticate with user's email address and access Sent Emails folder using Gmail API
def authenticate_gmail(folder_label='SENT'):
    try:
        # Get credentials from environment variables
        credentials_json = os.environ.get('GMAIL_CREDENTIALS_JSON')
        credentials_bytes = base64.b64decode(credentials_json)
        credentials = google.oauth2.credentials.Credentials.from_authorized_user_info(
            google.auth.transport.requests.Request(), 
            json.loads(credentials_bytes)
        )

        # Create a Gmail API client
        service = build('gmail', 'v1', credentials=credentials)

        # Get the user's Sent Emails folder ID
        query = f'label:{folder_label}'
        response = service.users().messages().list(userId='me', q=query).execute()
        folder_id = response['messages'][0]['labelIds'][0]

        # Get the emails in the Sent Emails folder
        emails = []
        page_token = None
        while True:
            response = service.users().messages().list(
                userId='me',
                q=query,
                labelIds=[folder_id],
                pageToken=page_token
            ).execute()
            emails.extend(response['messages'])
            page_token = response.get('nextPageToken')
            if not page_token:
                break

        # Get the full email data for each email in the Sent Emails folder
        email_data = []
        for email_obj in emails:
            email_id = email_obj['id']
            message = service.users().messages().get(userId='me', id=email_id).execute()
            email_data.append(message)

        return email_data
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

# Preprocess the email data
def preprocess_email_data(email_data, service, folder_label='SENT'):
    nlp = spacy.load("en_core_web_sm")
    email_data_preprocessed = []
    for email_obj in email_data:
        # Get the email thread ID
        thread_id = email_obj['threadId']
        
        # Get the full email data for the email thread
        response = service.users().threads().get(userId='me', id=thread_id, format='full').execute()
        
        # Get the label IDs for the email thread
        label_ids = response['messages'][0]['labelIds']
        
        # Check if the email thread is in user's Sent Emails folder
        if f'label:{folder_label}' not in label_ids:
            continue
        
        # Preprocess each email in the thread
        for message in response['messages']:
            email_data_preprocessed.extend(preprocess_message_parts(message))
    
    return email_data_preprocessed

def preprocess_message_parts(message):
    # Define the preprocess_text function
    def preprocess_text(text):
        # Decode the base64-encoded text
        text = base64.urlsafe_b64decode(text).decode('utf-8')
        
        # Parse the email message
        message = email.message_from_string(text)

# Extract the email body
email_body = ''
if message.is_multipart():
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            email_body = part.get_payload()
else:
    email_body = message.get_payload()

# Return the email body text
return email_body
def preprocess_message_parts(message):
    # Define the preprocess_text function
    def preprocess_text(text):
        # Decode the base64-encoded text
        text = base64.urlsafe_b64decode(text).decode('utf-8')
        
        # Parse the email message
        message = email.message_from_string(text)
        
        # Extract the email body
        email_body = ''
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    email_body = part.get_payload()
        else:
            email_body = message.get_payload()
            
        # Return the email body text
        return email_body
    
    preprocessed_parts = []
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            preprocessed_parts.extend(preprocess_message_parts(part))
    else:
        preprocessed_parts.append(preprocess_text(message['payload']['body']['data']))
    
    # Extract the email sender, recipients, subject, and timestamp
    sender = message['From']
    recipients = message['To']
    subject = message['Subject']
    timestamp = message['Date']
    
    # Concatenate the email metadata
    metadata = f'Sender: {sender}\nRecipients: {recipients}\nSubject: {subject}\nTimestamp: {timestamp}\n'
    
    # Tokenize and filter the email body text
    body_text = " ".join([preprocess_text(part) for part in preprocessed_parts])
    doc = nlp(body_text)
    filtered_text = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # Return the concatenated metadata and filtered email body text
    return metadata + " ".join(filtered_text)

# Train a language model on preprocessed email data
def train_language_model(email_data_preprocessed, model_name='bert-base-uncased', num_train_samples=None):
    if not num_train_samples:
        num_train_samples = len(email_data_preprocessed)
    train_data = random.sample(email_data_preprocessed, min(num_train_samples, len(email_data_preprocessed)))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=lambda data: {'input_ids': tokenizer(data, padding=True, truncation=True)['input_ids']},
    )

    trainer.train()

    # Save the trained language model to a file
    tokenizer.save_pretrained('./language_model')
    model.save_pretrained('./language_model')


# Query the language model using text-based questions
def query_language_model(question, model_path='./language_model', tokenizer_path='./language_model'):
    # Load the trained language model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Generate a response to the question using the language model
    fill_mask_pipeline = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    response = fill_mask_pipeline(question)[0]['sequence']

    return response.strip()


# Save the language model vector database in Pinecone
def save_language_model_to_pinecone(email_data_preprocessed, index_name, model_path='./language_model', tokenizer_path='./language_model', dimension=768):
    # Connect to Pinecone
    pinecone.init(api_key='<your-api-key>')

    # Check if the index already exists and delete it if it does
    if pinecone.list_indexes(index_name):
        pinecone.delete_index(index_name)

    # Create a new index in Pinecone to store our language model vectors
    pinecone.create_index(index_name=index_name, dimension=dimension)

    # Load the trained language model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Add the language model vectors to the index
    for text in email_data_preprocessed:
        vector = model(**tokenizer(text, return_tensors="pt")).last_hidden_state.mean(dim=1).tolist()[0]
        pinecone.add_items(index_name=index_name, data=[vector], item_ids=[str(email_data_preprocessed.index(text))])

    # Disconnect from Pinecone
    pinecone.deinit()


# Complete workflow for training the language model on a user's Sent Emails folder
def train_language_model_on_sent_emails(folder_label='SENT', model_name='bert-base-uncased', num_train_samples=None, index_name='<your-index-name>', dimension=768):
    try:
        # Authenticate with the user's email address and access Sent Emails folder using Gmail API
        credentials_json = os.environ.get('GMAIL_CREDENTIALS_JSON')
        credentials_bytes = base64.b64decode(credentials_json)
        credentials = google.oauth2.credentials.Credentials.from_authorized_user_info(
            google.auth.transport.requests.Request(), 
            json.loads(credentials_bytes)
        )

        # Create a Gmail API client
        service = build('gmail', 'v1', credentials=credentials)

        # Get the user's Sent Emails folder ID
        query = f'label:{folder_label}'
        response = service.users().messages().list(userId='me', q=query).execute()
        folder_id = response['messages'][0]['labelIds'][0]

        # Get the emails in the Sent Emails folder
        emails = []
        page_token = None
        while True:
            response = service.users().messages().list(
                userId='me',
                q=query,
                labelIds=[folder_id],
                pageToken=page_token
            ).execute()
            emails.extend(response['messages'])
            page_token = response.get('nextPageToken')
            if not page_token:
                break

        # Get the full email data for each email in the Sent Emails folder
        email_data = []
        for email_obj in emails:
            email_id = email_obj['id']
            message = service.users().messages().get(userId='me', id=email_id).execute()
            email_data.append(message)

        # Preprocess the email data
        email_data_preprocessed = preprocess_email_data(email_data, service, folder_label)

        # Train a language model on preprocessed email data
        train_language_model(email_data_preprocessed, model_name=model_name, num_train_samples=num_train_samples)

        # Save the trained language model to Pinecone
        save_language_model_to_pinecone(email_data_preprocessed, index_name=index_name, model_path='./language_model', tokenizer_path='./language_model', dimension=dimension)
    except HttpError as error:
        print(f'An error occurred: {error}')