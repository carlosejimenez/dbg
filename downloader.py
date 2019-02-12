import os
import requests
import json


if __name__ == '__main__':
    headers = {
        'X-DBP-APIKEY': 'e6e8d13f-2e66-476d-b375-c55b33eb7f8a',
    }

    params = (
        ('date', '2018-06-28'),
    )

    url = 'https://api.developer.deutsche-boerse.com/prod/xetra-public-data-set/1.0.0/xetra'

    response = requests.get(url, headers=headers, params=params)

    response = response.json()
