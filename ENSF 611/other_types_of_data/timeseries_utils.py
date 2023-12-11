from sklearn.metrics import get_scorer
import pandas as pd
import matplotlib.pyplot as plt

def create_xticks(data_index, freq='D', fmt_str="%a %m-%d"):
    """ create xtick_loc, xtick_str used for plt.xticks()
    
        returns (tuple[list[int], list[str]]): xtick_loc, xtick_str
    """
    pd_range = pd.date_range(start=data_index.min(), end=data_index.max(), freq=freq)
    
    # find the date locations in the original inded
    xtick_loc = [data_index.get_loc(dt) for dt in pd_range]
    # format the dates
    xtick_str = pd_range.strftime(fmt_str)
    
    return xtick_loc, xtick_str

def time_split(features, target, split_idx):
     # split the given features into a training and a test set
    X_train, X_test = features[:split_idx], features[split_idx:]
    # also split the target array
    y_train, y_test = target[:split_idx], target[split_idx:]
    
    return X_train, X_test, y_train, y_test

def plot_time_prediction(y_train, y_train_pred, y_val, y_val_pred, x_str="", y_str="", xticks=None):
    
    n_train, n_val = len(y_train), len(y_val)
    
    plt.figure(figsize=(10, 3))

    if xticks is not None:
        xticks_loc, xticks_str = xticks
        plt.xticks(xticks_loc, xticks_str, rotation=90,ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, n_train + n_val), y_val, '-', label="val")
    
    plt.plot(range(n_train), y_train_pred, '--', label="train pred")
    plt.plot(range(n_train, n_train + n_val), y_val_pred, '--', label="val pred")
    
    plt.legend(loc=(1.01, 0))
    plt.xlabel(x_str)
    plt.ylabel(y_str)

def print_scores(regressor, X_train, y_train, X_val, y_val, scorer_str='r2'):
    scorer = get_scorer(scorer_str)
    print(f"Train-set {scorer_str}: {scorer(regressor, X_train, y_train):.2f}")
    print(f"Val-set {scorer_str}: {scorer(regressor, X_val, y_val):.2f}")

def eval_on_features(features, target, regressor, n_val=64, x_str="Date", y_str="Rentals", xticks=None):
    """function to evaluate and plot a regressor on a given feature set
    """
    
    X_train, X_val, y_train, y_val = time_split(features, target, len(features)-n_val)
    
    regressor.fit(X_train, y_train)
    
    print_scores(regressor, X_train, y_train, X_val, y_val)
    print_scores(regressor, X_train, y_train, X_val, y_val, scorer_str='neg_root_mean_squared_error')
       
    y_train_pred = regressor.predict(X_train)
    y_val_pred = regressor.predict(X_val)
    
    if xticks is None:
        # get xticks from feature series index
        xticks = create_xticks(features.index)     


    
    plot_time_prediction(y_train, y_train_pred, y_val, y_val_pred, 
                         x_str=x_str, y_str=y_str, xticks=xticks)
    
def plot_feature_importances(feature_importances, feature_names):
    
    assert len(feature_importances) == len(feature_names), "Need to have as many importances as names"
    
    n_features = len(feature_importances)
    plt.barh(range(n_features), feature_importances, align='center')
    plt.yticks(range(n_features),feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features);
    
def plot_coefficients(coefficients, feature_names):
    
    assert len(coefficients) == len(feature_names), "Need to have as many coefficients as names"
    
    plt.plot(coefficients,'o')
    plt.xticks(range(len(coefficients)), feature_names, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.grid();
                                                  