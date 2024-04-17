from locust import HttpUser, between, task

class MyUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"
    @task
    def predict_image(self):
        # Load an image file to send in the request
        with open('Test_folder/1_4.png', 'rb') as f:
            files = {'file': ('1_4.png', f, '1_4/png')}
            self.client.post("/predict", files=files)
