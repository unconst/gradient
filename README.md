
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

This will install all the necessary packages, including `bittensor`, `torch`, `boto3`, and `transformers`, as specified in the `requirements.txt` file.

## Running
m
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

### Step 4: Configure Your Environment
Open a terminal and run the following commands to add your AWS credentials to your `~/.bash_profile`. Replace `YOUR_ACCESS_KEY_ID` and `YOUR_SECRET_ACCESS_KEY` with the credentials obtained in the previous step.
```bash
echo "export AWS_ACCESS_KEY_ID='AKIA3TN4TF2QQ4KC4CBA'" >> ~/.bash_profile
echo "export AWS_SECRET_ACCESS_KEY='a/VLFo0RIlS6WSn2BoLffsRW1frmwm5AyoFcQj2e'" >> ~/.bash_profile
```
### How to Run a Miner

To run a miner, you will need to execute the `miner.py` script, which is responsible for the mining process. This script handles the training of the model on your local machine, calculates the delta (the difference between the newly trained model and the master model), and then pushes this delta back to the network.

Here are the steps to run a miner:

1. Ensure you have followed the installation instructions and have all the necessary dependencies installed.

2. Open a terminal and navigate to the directory where `miner.py` is located.

3. Run the miner by executing the following command:
   ```bash
   python miner.py --uid <unique_identifier>
   ```
   Replace `<unique_identifier>` with a unique ID for your miner. This ID is used to track the contributions from your miner to the network.

4. The miner will start training the model locally. Progress will be shown in the terminal, including the loss after each epoch and a success message when a delta is pushed successfully.

5. To stop the miner, simply interrupt the process in your terminal (e.g., by pressing `Ctrl+C`).

By running a miner, you are contributing to the decentralized training of the model, helping improve its performance for everyone in the network.

### How to Run a Miner
### How to Run a Validator

Running a validator involves evaluating the deltas (changes) made by miners to the model and scoring them based on their impact on the model's performance. Validators play a crucial role in ensuring the quality and integrity of updates to the model. Follow the steps below to run a validator:

1. Ensure you have followed the installation instructions and have all the necessary dependencies installed.

2. Open a terminal and navigate to the directory where `validator.py` is located.

3. Run the validator by executing the following command:
   ```bash
   python validator.py --device <device> --bs <batch_size> --sl <sequence_length> --pages_per_epoch <pages_per_epoch>
   ```
   - Replace `<device>` with the device you want to use for computations (e.g., `cpu` or `cuda`).
   - Replace `<batch_size>` with the batch size for processing (e.g., `1`).
   - Replace `<sequence_length>` with the sequence length for each batch (e.g., `512`).
   - Replace `<pages_per_epoch>` with the number of pages to process per epoch (e.g., `3`).

4. The validator will start evaluating the deltas by applying them to the master model, computing the loss before and after applying each delta, and scoring them based on the change in loss.

5. Scores for each delta will be updated and logged, providing insights into which deltas improve the model's performance.

By running a validator, you are contributing to the decentralized evaluation of model updates, helping ensure that only beneficial changes are accepted into the master model.
