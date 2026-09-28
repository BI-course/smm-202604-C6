import requests

# PART 1: GET -- retrieving data from a public, read-only API

# Go to: https://publicapis.io/apis
# Go to: https://api.stackexchange.com/docs

print("PART 1: GET request (retrieve data)\n")

# URL Breakdown
# Base URL: https://api.stackexchange.com/2.3/ — The root address and API version.
# Endpoint: questions — The specific resource or collection you are requesting data from.
# Query Parameters: ?order=desc&sort=activity&pagesize=3&site=stackoverflow — The filters applied to sort and target the Stack Overflow site.

response = requests.get('https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&pagesize=3&site=stackoverflow')

for data in response.json()['items']:
    print('=' * 50)
    print('Title: ', data['title'])
    print('Link: ', data['link'])
    print('Answer Count: ', data['answer_count'])
    print('=' * 50, '\n')


# PART 2: POST -- sending data to an API, with headers

# Go to: https://jsonplaceholder.typicode.com/guide/

# https://jsonplaceholder.typicode.com is a free fake REST API built
# specifically for learning and testing. Nothing you send here is
# actually saved on a server -- it is safe to run this as many times
# as you like.

print("\nPART 2: POST request (create data), with headers\n")

create_url = 'https://jsonplaceholder.typicode.com/posts'

new_post = {
    'title': 'Learning APIs in BBT 4206',
    'body': 'This is a test post created from a Python script.',
    'userId': 1,
}

headers = {
    'Content-Type': 'application/json',  # tells the server the payload is JSON
    'Accept': 'application/json',        # tells the server we expect JSON back
}

response = requests.post(create_url, json=new_post, headers=headers)

print('Status Code:', response.status_code)  # This should be 201 = Created
print('Response Body:', response.json())


# PART 3: PUT -- updating existing data (idempotent)

# Recall from the lecture: PUT replaces a resource. Calling it again
# with the same payload produces the same end state -- that is what
# "idempotent" means.

print("\nPART 3: PUT request (update data)\n")

update_url = 'https://jsonplaceholder.typicode.com/posts/1'

updated_post = {
    'id': 1,
    'title': 'Updated Title',
    'body': 'This post has been updated.',
    'userId': 1,
}

response = requests.put(update_url, json=updated_post, headers=headers)

print('Status Code:', response.status_code)  # This should be 200 = OK
print('Response Body:', response.json())


# PART 4: DELETE -- removing data (idempotent)

print("\nPART 4: DELETE request (remove data)\n")

delete_url = 'https://jsonplaceholder.typicode.com/posts/1'

response = requests.delete(delete_url)

print('Status Code:', response.status_code)  # This should be 200 = OK (deleting again would still be 200)


# PART 5: GET -- comparing historical rainfall (El Nino vs. a normal year)

# Go to: https://open-meteo.com/en/docs/historical-weather-api

# Open-Meteo is a free weather API that needs no API key. Its archive
# endpoint returns historical daily weather for any coordinates, based on
# reanalysis data going back to 1940 -- This enables us to compare Nairobi's
# rainfall during the Oct-Dec 1997 El Nino "short rains" season (one of
# the strongest El Nino events on record, associated with severe flooding
# across East Africa) against the Oct-Dec 2025 short rains.

print("\nPART 5: GET request (compare historical rainfall)\n")

archive_url = 'https://archive-api.open-meteo.com/v1/archive'

nairobi_params = {
    'latitude': -1.2921,
    'longitude': 36.8219,
    'daily': 'precipitation_sum',
    'timezone': 'Africa/Nairobi',
}

seasons = {
    'Oct-Dec 1963 Short Rains': ('1963-10-01', '1963-12-31'),
    'El Nino (Oct-Dec 1997)': ('1997-10-01', '1997-12-31'),
    'Oct-Dec 2020 Short Rains': ('2020-10-01', '2020-12-31'),
    'Oct-Dec 2021 Short Rains': ('2021-10-01', '2021-12-31'),
    'Oct-Dec 2022 Short Rains': ('2022-10-01', '2022-12-31'),
    'Oct-Dec 2023 Short Rains': ('2023-10-01', '2023-12-31'),
    'Oct-Dec 2024 Short Rains': ('2024-10-01', '2024-12-31'),
    'Oct-Dec 2025 Short Rains': ('2025-10-01', '2025-12-31'),
    # 'Mar-May 2026 Long Rains': ('2026-03-01', '2026-05-31'),
}

# seasons.items() gives us each dictionary entry as a (key, value) pair.
# Here the value itself is a tuple -- (start_date, end_date) -- so Python
# unpacks it directly into two separate variables in the same line.
# On the first loop: label = 'El Nino (Oct-Dec 1997)', start_date =
# '1997-10-01', end_date = '1997-12-31'. On the second loop, the values
# for 2020 are unpacked the same way.
for label, (start_date, end_date) in seasons.items():

    # {**nairobi_params, ...} creates a NEW dictionary by first copying
    # every key-value pair out of nairobi_params (the ** "unpacks" it),
    # then adding two more keys: 'start_date' and 'end_date'. This means
    # we do not have to retype latitude/longitude/daily/timezone for
    # every season -- we reuse the shared settings and only change what's
    # different (the dates) each time through the loop.
    params = {**nairobi_params, 'start_date': start_date, 'end_date': end_date}

    # requests.get() sends an HTTP GET request. Passing a dictionary as
    # `params` tells the requests library to automatically convert it
    # into a URL query string for us (e.g., ?latitude=-1.2921&longitude=
    # 36.8219&start_date=1997-10-01&...), so we never have to build that
    # string by hand.
    response = requests.get(archive_url, params=params)

    # The JSON response has a "daily" section containing several parallel
    # lists -- one entry per day in the requested range. We only asked
    # for 'precipitation_sum', so we "drill down" (a concept from Module 1 in
    # BI1) to that specific list:
    # response.json() -> the whole JSON body as a Python dict
    #   ['daily'] -> the dict of daily data series
    #     ['precipitation_sum'] -> the list of daily rainfall totals (mm)
    daily_rainfall = response.json()['daily']['precipitation_sum']

    # Some days can come back as None (null in JSON) if a value is
    # missing from the underlying weather data for that date. Adding
    # None to a number would raise a TypeError, so this generator
    # expression only includes a day's value (`mm`) in the sum if it is
    # not None, then sum() adds up everything that survives the filter.
    total_rainfall = sum(mm for mm in daily_rainfall if mm is not None)

    print('=' * 50)
    print(label)
    print('Total rainfall (mm):', round(total_rainfall, 1))
    print('=' * 50, '\n')
