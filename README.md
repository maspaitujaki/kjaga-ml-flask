# KJaga ML Model
This service use the following tech stacks:
1. Python
2. PostgreSQL
3. Google's Cloud Storage

## Getting Started

First, install the dependencies the development server:
```bash
pip install -r requirements.txt
```
Create a .env file, look at .env.example for the format. These are the description for each values:
| Value  | Description  | 
| :------------ |:---------------| 
| PG_HOST | Host of the database. e.g localhost, Ip address |
| PG_USER | Postgres username |
| PG_PASSWORD | Postgres password |
| PG_DATABASE | Database Name |

notes: The database should be the same as in kjaga-backend

Run the development server:
```bash
flask run
```

### Deployment
Use the Dockerfile provided to deliver the service, in this case we use Cloud Build, Artifact Registry, and Cloud Run
