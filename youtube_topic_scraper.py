
from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm
import time
import re

API_KEY = "AIzaSyCUVsiifkkmDv97R4fd7qVAjgNzsYGnZK8"

youtube = build('youtube', 'v3', developerKey=API_KEY)

# 🎯 Topics + search queries
TOPICS = {
    "AI": "AI news debate interview",
    "War": "war news analysis debate",
    "Stock": "stock market news india",
    "Economy": "inflation economy news",
    "Tech": "latest technology news"
}

TARGET_PER_TOPIC = 3000  # 5 topics → ~7500 total

all_data = []


# ❌ Remove unwanted videos
def is_good_video(title):
    title = title.lower()
    
    bad_words = [
        "trailer", "song", "music", "movie",
        "teaser", "full movie", "lyrics"
    ]
    
    for word in bad_words:
        if word in title:
            return False
    
    return True


# ✅ Filter comments
def is_valid_comment(text):
    if len(text.split()) < 4:
        return False
    
    if re.fullmatch(r'[\W_]+', text):
        return False
    
    return True


# 🔍 Search videos for topic
def search_videos(query):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=20
    )
    return request.execute()['items']


# 💬 Get comments
def get_comments(video_id, topic):
    comments = []
    next_page_token = None

    while len(comments) < 300:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']

                text = snippet['textDisplay']
                likes = snippet['likeCount']

                if is_valid_comment(text):
                    comments.append([text, topic, likes])

                if len(comments) >= 300:
                    break

            next_page_token = response.get('nextPageToken')

            if not next_page_token:
                break

        except:
            break

    return comments


# 🚀 MAIN
for topic, query in TOPICS.items():
    print(f"\n🔍 Searching videos for {topic}...")

    videos = search_videos(query)

    topic_count = 0

    for video in videos:
        title = video['snippet']['title']
        video_id = video['id']['videoId']

        if not is_good_video(title):
            continue

        print(f"📺 {title[:50]}...")

        comments = get_comments(video_id, topic)

        all_data.extend(comments)
        topic_count += len(comments)

        print(f"✅ +{len(comments)} comments")

        time.sleep(1)

        if topic_count >= TARGET_PER_TOPIC:
            break


# 📊 Save dataset
df = pd.DataFrame(all_data, columns=['text', 'topic', 'likes'])
df.to_csv("clean_dataset_8000.csv", index=False)

print("\n🎉 FINAL CLEAN DATASET CREATED")
print("📊 Total rows:", len(df))