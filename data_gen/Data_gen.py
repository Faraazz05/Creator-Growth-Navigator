import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "C:\\Users\\Faraz\\Documents\\Creator Growth Navigator\\data\\processed"
# -----------------------------
FILENAME = "creator_daily_metrics_730d.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, FILENAME)

USERNAME = "creator_alpha"
SEED = 123
np.random.seed(SEED)

N_DAYS = 730
START_DATE = datetime(2023, 1, 1)

# Starting scale
START_FOLLOWERS = 120_000  # >100,000 as requested

# Typical patterns and ranges
USUAL_POST_TIMES = ['08:00', '12:30', '18:30', '21:00']
HASHTAGS_RANGE = (4, 22)

# Seasonality profiles (monthly multipliers for engagement/growth)
MONTH_SEASONALITY = {
    1: 0.95, 2: 0.98, 3: 1.02, 4: 1.05, 5: 1.06, 6: 1.08,
    7: 1.10, 8: 1.07, 9: 1.03, 10: 1.00, 11: 1.04, 12: 1.15
}

# Content preference weights (affect engagement)
CONTENT_WEIGHTS = {"post": 1.0, "story": 0.6, "reel": 1.5}

def safe_div(a, b):
    return a / b if b != 0 else 0.0

# Ensure output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []
followers = START_FOLLOWERS
recent_posts = []  # for 7-day variance (consistency score)

for i in range(N_DAYS):
    date = START_DATE + timedelta(days=i)
    month_mult = MONTH_SEASONALITY[date.month]

    # Posting behavior (seasonally modulated)
    posts = np.random.poisson(1.2 * month_mult)
    stories = np.random.poisson(3.0 * month_mult)
    reels = np.random.binomial(2, 0.45)  # 0–2 reels
    ads_posted = np.random.binomial(1, 0.10)

    # Hashtags and posting time
    hashtags_used = posts * np.random.randint(*HASHTAGS_RANGE)
    avg_hashtag_count = safe_div(hashtags_used, posts)
    post_time = np.random.choice(USUAL_POST_TIMES)

    # Engagement scaling (diminishing returns with size)
    quality_noise = np.random.uniform(0.85, 1.15)
    follower_scale = (followers ** 0.65) / (START_FOLLOWERS ** 0.65)

    # Engagement by type
    post_likes = int(posts * 350 * month_mult * quality_noise * CONTENT_WEIGHTS["post"] * follower_scale)
    reel_plays = int(reels * 2200 * month_mult * quality_noise * CONTENT_WEIGHTS["reel"] * follower_scale)
    reel_engagement = int(reels * 280 * month_mult * quality_noise * CONTENT_WEIGHTS["reel"] * follower_scale)
    story_reach = int(stories * 900 * month_mult * quality_noise * CONTENT_WEIGHTS["story"] * follower_scale)
    story_engagement = int(stories * 120 * month_mult * quality_noise * CONTENT_WEIGHTS["story"] * follower_scale)

    # Reach and visits
    organic_reach = int(followers * np.random.uniform(0.3, 0.9) * month_mult)
    activity_reach = posts * 450 + reels * 700 + stories * 120
    reach = int((organic_reach + activity_reach) * np.random.uniform(0.95, 1.10))
    ad_reach_boost = int(ads_posted * np.random.uniform(2_000, 10_000) * month_mult)
    reach += ad_reach_boost

    profile_visits = int(reach * np.random.uniform(0.02, 0.08))
    saves = int(posts * np.random.uniform(25, 120) * month_mult * follower_scale)
    shares = int(posts * np.random.uniform(18, 85) * month_mult * follower_scale)
    comments = int(posts * np.random.uniform(35, 180) * month_mult * follower_scale)

    total_interactions = post_likes + reel_engagement + story_engagement + saves + shares + comments
    engagement_rate = safe_div(total_interactions, max(reach, 1))
    ad_engagement = int(ads_posted * np.random.uniform(120, 480) * month_mult)

    # Follows/unfollows engine
    baseline_follows = int(engagement_rate * reach * np.random.uniform(0.005, 0.018))
    content_bonus = int((posts * 6 + reels * 14 + stories * 3) * np.random.uniform(0.8, 1.3))
    ad_bonus = int(ads_posted * np.random.uniform(40, 200))
    went_viral = np.random.rand() < 0.02
    viral_boost = int(went_viral * np.random.uniform(800, 7000) * month_mult)
    follows = baseline_follows + content_bonus + ad_bonus + viral_boost

    unfollows = int(np.random.poisson(lam=max(5, followers * 0.00015)) + ads_posted * np.random.randint(0, 40))
    growth = follows - unfollows
    followers = max(0, followers + growth)

    # Competitor cadence (for benchmarking UI)
    comp_posts = np.random.poisson(1.3 * month_mult)
    comp_reels = np.random.binomial(2, 0.40)
    comp_stories = np.random.poisson(2.8 * month_mult)

    # Timing feature
    optimal_hours = {'08:00', '18:30', '21:00'}
    posted_in_optimal_window = 1 if post_time in optimal_hours else 0

    # Content mix shares
    total_content = posts + reels + stories
    share_posts = safe_div(posts, total_content)
    share_reels = safe_div(reels, total_content)
    share_stories = safe_div(stories, total_content)

    # Consistency (7d variance)
    recent_posts.append(posts)
    if len(recent_posts) > 7:
        recent_posts.pop(0)
    post_consistency_variance_7d = float(np.var(recent_posts)) if len(recent_posts) >= 2 else 0.0

    # Saturation signal
    saturation_flag = 1 if (posts + reels >= 5 and engagement_rate < 0.05) else 0

    # ROI (minutes and follows per hour)
    minutes_per_post = 45
    minutes_per_reel = 120
    minutes_per_story = 8
    minutes_spent = posts * minutes_per_post + reels * minutes_per_reel + stories * minutes_per_story
    roi_follows_per_hour = safe_div(follows, (minutes_spent / 60) if minutes_spent > 0 else 1)

    row = {
        "username": USERNAME,
        "date": date.strftime("%d-%m-%Y"),

        # Outcomes
        "followers": followers,
        "growth": growth,
        "follows": follows,
        "unfollows": unfollows,

        # Activity
        "posts": posts,
        "stories": stories,
        "reels": reels,
        "ads_posted": ads_posted,
        "post_time": post_time,
        "posted_in_optimal_window": posted_in_optimal_window,

        # Hashtags
        "hashtags_used": hashtags_used,
        "avg_hashtag_count": avg_hashtag_count,

        # Engagement and reach
        "reach": reach,
        "profile_visits": profile_visits,
        "post_likes": post_likes,
        "reel_plays": reel_plays,
        "reel_engagement": reel_engagement,
        "story_reach": story_reach,
        "story_engagement": story_engagement,
        "comments": comments,
        "saves": saves,
        "shares": shares,
        "engagement_rate": engagement_rate,
        "ad_engagement": ad_engagement,

        # Mix and diagnostics
        "share_posts": share_posts,
        "share_reels": share_reels,
        "share_stories": share_stories,
        "post_consistency_variance_7d": post_consistency_variance_7d,
        "saturation_flag": saturation_flag,

        # Competitor benchmarks (for UI comparisons)
        "comp_posts": comp_posts,
        "comp_reels": comp_reels,
        "comp_stories": comp_stories,

        # ROI
        "minutes_spent": minutes_spent,
        "roi_follows_per_hour": roi_follows_per_hour,

        # Seasonality markers
        "month": date.month,
        "quarter": (date.month - 1) // 3 + 1,
        "went_viral": int(went_viral)
    }

    rows.append(row)

# Build DataFrame and write CSV
df = pd.DataFrame(rows)

# Enforce dtypes
int_cols = [
    "followers","growth","follows","unfollows","posts","stories","reels","ads_posted",
    "hashtags_used","reach","profile_visits","post_likes","reel_plays","reel_engagement",
    "story_reach","story_engagement","comments","saves","shares",
    "posted_in_optimal_window","saturation_flag","comp_posts","comp_reels","comp_stories",
    "month","quarter","went_viral","minutes_spent"
]
for c in int_cols:
    df[c] = df[c].astype(int)

float_cols = [
    "avg_hashtag_count","engagement_rate","share_posts","share_reels","share_stories",
    "post_consistency_variance_7d","roi_follows_per_hour"
]
for c in float_cols:
    df[c] = df[c].astype(float)

# Save CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} rows to: {OUTPUT_PATH}")
