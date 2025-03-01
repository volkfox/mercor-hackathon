import os
from getpass import getpass

def setup_env():
    """Create a .env file with the Anthropic API key."""
    if os.path.exists(".env"):
        overwrite = input(".env file already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Setup cancelled.")
            return
    
    api_key = getpass("Enter your Anthropic API key: ")
    
    with open(".env", "w") as f:
        f.write(f"ANTHROPIC_API_KEY={api_key}\n")
    
    print(".env file created successfully!")
    print("Your API key is stored in the .env file and will be ignored by git.")

if __name__ == "__main__":
    setup_env() 