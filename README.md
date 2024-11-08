## Decovision

<img width="1713" alt="Screenshot 2024-11-08 at 6 05 07 PM" src="https://github.com/user-attachments/assets/230b70ae-fca0-4150-88eb-6bcff5ed5bd8">


## How to use

### 1. Clone this project's repository

In your Terminal app

- Type `git clone git@github.com:siegblink/interior-designer-ai.git`
- Or type `git clone https://github.com/siegblink/interior-designer-ai.git`

### 2. Install the project dependencies

Go to the project's directory

- Type `cd interior-designer-ai`
- Then, `npm install`

### 3. Create an account at [replicate](https://replicate.com/)

![create-account-in-replicate](public/create-account-in-replicate.png)

### 4. Create your API token and copy it

![image (2)](https://github.com/user-attachments/assets/90750066-207d-4801-ba6a-fa9043ff8a2c)


### 5. Rename the `.env.example` file to `.env.local`

### 6. In `.env.local`, replace the placeholder _your_api_token_ with your API token

```
# Replace 'your-api-token' with your own API token from replicate
REPLICATE_API_TOKEN=your_api_token
```

### 7. Run the project

Back in your Terminal in the project directory, type `npm run dev`

