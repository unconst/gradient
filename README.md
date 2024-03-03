
# Gradient: Incentivized Distributed Training

![System Design](/docs/systemdesign.jpg)

# Introduction
The subnet uses Bittensor's incentive system to reward computers for producing gradients which maximially reduce the loss on the model architecture hosted by the subnet owner. The use of S3 buckets factilitates high bandwidth communication between peers. 

## Table of Contents
- [Installation](#installation)
- [Running](#running)
  - [Setting Up AWS](#setting-up-aws)
    - [Step 1: Create an AWS Account](#step-1-create-an-aws-account)
    - [Step 2: Create an S3 Bucket](#step-2-create-an-s3-bucket)
    - [Step 3: Obtain AWS Access Credentials](#step-3-obtain-aws-access-credentials)
    - [Step 4: Configure Your Environment Using a `.env` File](#step-4-configure-your-environment-using-a-env-file)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)

## Installation
To install the necessary dependencies for this project, please ensure you have Python 3.6 or higher installed on your system. Then, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/YumaRao/gradient.git
   ```
2. Navigate to the cloned repository directory:
   ```bash
   cd gradient
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the repository
   ```bash
   python3 -m pip install -e.
   ```

## Running
Before running a miner or validator, you need to set up an S3 bucket on AWS and configure your environment to use AWS credentials. Follow these steps to get started:

### Step 1: Create an AWS Account
If you don't already have an AWS account, go to the AWS homepage and sign up.

### Step 2: Create an S3 Bucket
1. Log in to the AWS Management Console and open the Amazon S3 console at https://console.aws.amazon.com/s3/.
2. Click **Create bucket**.
3. Provide a unique name for your bucket and select the AWS Region where you want the bucket to reside.
4. Follow the on-screen instructions to configure options and set permissions. For most use cases, the default settings will suffice.
5. Click **Create bucket** to finalize.

### Step 3: Obtain AWS Access Credentials
1. Navigate to the IAM console at https://console.aws.amazon.com/iam/.
2. In the navigation pane, choose **Users**, then **Add user**.
3. Enter a user name, select **Programmatic access** for the AWS access type, and click **Next: Permissions**.
4. Choose **Attach existing policies directly** and select the **AmazonS3FullAccess** policy.
5. Follow the rest of the on-screen instructions to create the user. After the user is created, you'll be provided with an **Access key ID** and a **Secret access key**. Make a note of these credentials.

### Step 4: Configure Your Environment Using a `.env` File
To securely manage your AWS credentials, you will create a `.env` file in your project's root directory and add your credentials to it. Follow these steps:
1. Open a terminal and navigate to the root directory of your project.
2. Run the following commands to create a `.env` file and add your AWS credentials to it. Replace `YOUR_ACCESS_KEY_ID` and `YOUR_SECRET_ACCESS_KEY` with the credentials obtained in the previous step:
   ```bash
   touch .env
   echo "AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID" >> .env
   echo "AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY" >> .env
   ```

### How to Run a Miner

To run a miner, you will need to execute the `neurons/miner.py` script, which is responsible for the mining process. This script handles the training of the model on your local machine, calculates the delta (the difference between the newly trained model and the master model), and then pushes this delta back to the network.

Here are the steps to run a miner:
1. Ensure you have followed the installation instructions and have all the necessary dependencies installed.
3. Run the miner by executing the following command:
   ```bash
   python neurons/miner.py --device <device> --wallet.name <your wallet name> --wallet.hotkey <your wallet hotkey>
   ```
4. The miner will start training the model locally. Progress will be shown in the terminal, including the loss after each epoch and a success message when a delta is pushed successfully.
5. To stop the miner, simply interrupt the process in your terminal (e.g., by pressing `Ctrl+C`).

### How to Run a Validator

Running a validator involves evaluating the deltas (changes) made by miners to the model and scoring them based on their impact on the model's performance. Validators play a crucial role in ensuring the quality and integrity of updates to the model. Follow the steps below to run a validator:

1. Ensure you have followed the installation instructions and have all the necessary dependencies installed.
3. Run the validator by executing the following command:
   ```bash
   python validator.py --device <device> --wallet.name <your wallet name> --wallet.hotkey <your wallet hotkey>
   ```