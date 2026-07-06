
def run_model( model,X_train, y_train, X_val, y_val, X_test, y_test, nombre_modelo: str, nombre_features: str, *, feature_cols=None, plot_importances: bool = True,):
    #train_model + evaluate_model + evaluate_test 

    model = train_model(model, X_train, y_train)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    metrics_train_val = evaluate_model(
        model, X_train, y_train, X_val, y_val, nombre_modelo, nombre_features, ax=axes[0]
    )
    metrics_test = evaluate_test(model, y_train, X_test, y_test, nombre_features, ax=axes[1])
 
    plt.suptitle(f"{nombre_modelo} - {nombre_features}")
    plt.tight_layout()
    plt.show()
 
    metrics = pd.concat([metrics_train_val, metrics_test])
 
    underlying = _get_underlying_estimator(model)
 
    if plot_importances and feature_cols is not None and hasattr(underlying, "feature_importances_"):
        plot_feature_importances(model, feature_cols, nombre_features)
 
    return model, metrics


# train and evaluate
# separar en varios:
#train_model
#evaluate_model (sobre train y validation)
#evaluate_test
def run_model( model, X_train, y_train, X_val, y_val, X_test, y_test, nombre_modelo: str, nombre_features: str, *, feature_cols=None, plot_importances: bool = True,):
    print(f"{nombre_modelo} - {nombre_features}")


    emotion_labels = sorted(pd.unique(y_train))

    model.fit(X_train, y_train)

    metrics = pd.DataFrame(
        [
            evaluate_split(model, X_train, y_train, "train"),
            evaluate_split(model, X_val, y_val, "validation"),
            evaluate_split(model, X_test, y_test, "test"),
        ]
    ).set_index("split")

    print(metrics.round(4))

    print(f"\nClassification report - Validation ({nombre_features})")
    print(classification_report_df(model, X_val, y_val).round(3))

    print(f"\nClassification report - Test ({nombre_features})")
    print(classification_report_df(model, X_test, y_test).round(3))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_confusion_for_split(model, X_val, y_val, emotion_labels, "Validation", ax=axes[0])
    plot_confusion_for_split(model, X_test, y_test, emotion_labels, "Test", ax=axes[1])

    plt.suptitle(f"{nombre_modelo} - {nombre_features}")
    plt.tight_layout()
    plt.show()

    underlying = _get_underlying_estimator(model)

    if plot_importances and feature_cols is not None and hasattr(underlying, "feature_importances_"):
        importances = top_feature_importances(underlying, feature_cols, top_n=20)
        print(importances)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importances, x="importance", y="feature", color="#2a9d8f")
        plt.title(f"Top 20 Feature Importances ({nombre_features})")
        plt.tight_layout()
        plt.show()

    return model, metrics

