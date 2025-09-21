def get_ranking(recommender, user, items_set_B):
    """Get the ranking of items for a given user and model."""
    return recommender.predict([user], items_set_B)[0]