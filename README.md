
Prerequisites

Python (Version 3.9 to 3.13)
Download Python from: https://www.python.org/downloads/
During installation, make sure to check the box that says "Add Python to PATH".

To verify the installation, open Command Prompt and run:
python --version
If the output looks like: Python 3.11.0, the installation is successful.

Groq API Key
Generate your API key at: https://console.groq.com/keys

How to get the key:

Go to the Groq Console.

Click the "Create API Key" button.

Copy and save the key. It will be displayed only once.

Installation:

Download the Chatbot.zip file to your Windows machine and unzip it.

Add your Groq API key to the .streamlit/secrets.toml file.

Run the run_chatbot.bat file.

Note: The first run may take up to 15 minutes to download and install all required dependencies.

Usage

Open a PDF, XLSX or DOCX file and ask questions about its content.

If you are not satisfied with the model’s answers, you can fine-tune the response quality by adjusting parameters:

Change the temperature value (line 55)

Change the top-k value (line 145)
