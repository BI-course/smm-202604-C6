# Marking Guide (Out of 100)

This is how your submission will be marked. Every row is checked by **what your API and video actually show working** -- not by which dataset or algorithm you used. Use this guide to check your own work before you submit.

| #                   | What you need to show                                                                                                             |   Marks |
|---------------------|:----------------------------------------------------------------------------------------------------------------------------------|--------:|
| 1                   | Your **regression** endpoint returns a correct prediction, using a model loaded from a saved file (not retrained inside `api.py`) |      20 |
| 2                   | Your **classification** endpoint returns a correct prediction, loaded from a saved file                                           |      20 |
| 3                   | Your **clustering** endpoint returns a correct prediction, loaded from a saved file                                               |      20 |
| 4                   | Your **recommender** (association rules) endpoint returns a correct recommendation, loaded from a saved file                      |      20 |
| 5                   | Your API is **Dockerized** and served with **Gunicorn** (not the Flask development server)                                        |       5 |
| 6                   | An **Nginx** reverse proxy correctly forwards requests to your containerized API                                                  |       5 |
| 7                   | Your **same, unmodified `api.py`** is deployed and reachable live on **Render**                                                   |      10 |
| **Subtotal so far** |                                                                                                                                   | **100** |

**You only need 3 of the 4 model rows (1-4) working to pass the Baseline tier -- the missing one is simply not awarded those marks, nothing is deducted for leaving it out.** All 4 working is required to reach Intermediate or Advanced.

## Bonus (Advanced tier only) -- up to +10 extra

| #  | What you need to show                                                                                     | Bonus marks |
|----|:----------------------------------------------------------------------------------------------------------|------------:|
| 8  | A **Hugging Face Space** (one Gradio app, one tab per model, all four give correct predictions)           |          +4 |
| 9  | A **Streamlit Community Cloud app** (one tab per model, all four give correct predictions)                |          +4 |
| 10 | A simple **HTML/CSS/JS web page** that demonstrates your API, with basic error handling for missing input |          +2 |

Your total is capped at 100 even if bonus marks would push you higher.

## What Gets Marks Taken Away

- **-10** if a model is retrained live inside `api.py` instead of loaded from a saved file. Loading from disk is the entire point of this lab -- an API that retrains on every request is not what you are being asked to build.
- **-10** if the public URL for your tier is missing **and** your video does not clearly show that deployment working instead.
- **-5** per team member whose individual contribution is not clearly documented at the top of your submission (which part they did, and a link to their branch).

## Before You Submit, Check:

- [ ] Every endpoint you claim works actually returns a correct answer when you test it fresh, right before recording your video
- [ ] Your video shows a real request and a real response for each endpoint you are claiming marks for -- not just the code, and not just the server starting up
- [ ] If you attempted Intermediate or Advanced, your video also shows the Dockerized/Nginx version and the Render URL responding from outside your own machine
- [ ] Every team member's contribution and branch link is filled in at the top of your submission
