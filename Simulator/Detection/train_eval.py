# 분할->학습->평가 모두 한 번에 처리하는 파일.
def split_data(df):
    # dataframe을 train, validation, test set으로 분할하는 함수.
    # train: 60%, validation: 20%, test: 20% 으로 분할.
    # dataset 자체를 분할하기 때문에, get_feature_columns() 등의 schema_utils 함수는 호출하지 않음.
    from sklearn.model_selection import train_test_split

    # train_val(train + validation)과 test set으로 분할.
    train_val, test = train_test_split(
        df,
        test_size = 0.2, # 20%를 test set으로 분할.
        random_state = 42, # 재현성 확보를 위해 random_state 고정.(값이 중요한 게 아님. 동일 값을 사용하는 게 중요.)
        stratify = df["is_phishing"]
    )

    # train_val(train + validation)을 train과 validation set으로 분할.
    train, validation = train_test_split(
        train_val,
        test_size = 0.25, # 25%를 validation set으로 분할 
        random_state = 42,
        stratify = train_val["is_phishing"]
    )   

    return train, validation, test
