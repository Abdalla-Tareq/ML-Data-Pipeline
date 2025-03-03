import os
import io
import pandas as pd
from tkinter import *
from tkinter import filedialog
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

main_window = Tk()
main_window.title("ML_Project")
main_window.geometry("400x600")
main_window.resizable(False, False)
main_window.configure(background="#08416B")

preprocessing_button = Button(main_window, text="Preprocessing", font=("Monaco", 15, "bold"), bg="#ffd343", width='31', height='3', command=lambda: open_preprocessing_window())
preprocessing_button.pack()
preprocessing_button.place(x=10, y=180)

classification_button = Button(main_window, text="Classification", font=("Monaco", 15, "bold"), bg="#ffd343", width='31', height='3', command=lambda: open_classification_window())
classification_button.pack()
classification_button.place(x=10, y=300)

clustering_button = Button(main_window, text="Clustering", font=("Monaco", 15, "bold"), bg="#ffd343", width='31', height='3', command=lambda : open_clustering_window())
clustering_button.pack()
clustering_button.place(x=10, y=420)

select_dataset_button = Button(main_window, text="Select DataSet", command=lambda: select_dataset())
select_dataset_button.config(bg="#ffd343", font=("Monaco", 12, "bold"))
select_dataset_button.pack()
select_dataset_button.place(x=130, y=30)

def select_dataset():
    global dataset_path
    dataset_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    data = pd.read_csv(dataset_path)
    print(data.head())
    if dataset_path:
        dataset_label = Label(main_window, text=dataset_path, font=("Monaco", 11), bg="#f0f4c3", wraplength=300, width=40, height=4)
        dataset_label.pack()
        dataset_label.place(x=20, y=70)

        info_button = Button(main_window, text="Info", font=("Monaco", 9, "bold"), bg="#84ffff", command=lambda: display_dataset_info(dataset_path))
        info_button.pack()
        info_button.place(x=175, y=148)

def display_dataset_info(file):
    info_window = Toplevel()
    info_window.title("DataSet Information")
    info_window.geometry("400x380")
    info_window.resizable(False, False)

    buffer = io.StringIO()
    pd.read_csv(file).info(buf=buffer)
    buffer.seek(0)
    data_info_string = buffer.read()

    text = Text(info_window, font=("Monaco", 10), bg="black", fg="#76ff03")
    text.pack()
    text.insert(END, data_info_string)
    text.config()

def close_window(window):
    window.destroy()
    main_window.deiconify()

def open_operation_window(operation_function):
    operation_window = Toplevel()
    operation_window.title(operation_function.__name__)
    operation_window.geometry("400x300")
    operation_window.resizable(True, True)
    operation_window.configure(background="#08416C")

    operation_function(operation_window)
##########################################################################################################################################
############################################################## Preprocessing #############################################################
##########################################################################################################################################

def open_preprocessing_window():
    main_window.withdraw()
    preprocess_window = Toplevel()
    preprocess_window.title("Preprocessing")
    preprocess_window.geometry("400x620")
    preprocess_window.resizable(False, False)
    preprocess_window.configure(background="#08416B")
    back_button = Button(preprocess_window, text="< Back", font=("Monaco", 12, "bold"), bg="#ffd343", width='8', command=lambda: close_window(preprocess_window))
    back_button.pack()
    back_button.place(x=20, y=580)

    operations = [
        ("Handle Missing Values", handle_missing_values_window),
        ("Normalize Data", normalize_data_window),
        ("Standardize Data", standardize_data_window),
        ("Encode Categorical Data", encode_categorical_data_window),
        ("Feature Selection Data", feature_selection_window),
        ("Handle Imbalanced Data", handle_imbalanced_data_window),
        ("Split Data", split_data_window)
    ]

    row = 0
    for operation_name, operation_function in operations:
        button = Button(preprocess_window, text=operation_name, font=("Monaco", 15, "bold"), bg="#ffd343", width='31', height='2',command=lambda op=operation_function: open_operation_window(op))
        button.grid(row=row, column=0, padx=10, pady=10)
        row += 1
################################################### handle_missing_values ####################################################
def handle_missing_values_window(window):
    columns_label = Label(window, text="Enter columns range (C1:C2):", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    strategy_label = Label(window, text="Select strategy:", font=("Monaco", 12), bg="#08416B", fg="white")
    strategy_label.pack(pady=10)

    strategy_var = StringVar(window)
    strategy_var.set("mean")
    strategy_menu = OptionMenu(window, strategy_var, "mean", "median", "most_frequent")
    strategy_menu.pack(pady=10)

    def handle_missing_values():
        dataset = pd.read_csv(dataset_path)
        start, end = map(int, columns_entry.get().split(":"))
        strategy = strategy_var.get()
        imputer = SimpleImputer(strategy=strategy)
        dataset.iloc[:, start:end] = imputer.fit_transform(dataset.iloc[:, start:end])
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_handle_missing_values" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"Imputed data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=handle_missing_values)
    execute_button.pack(pady=10)

################################################### normalize_data ####################################################
def normalize_data_window(window):
    columns_label = Label(window, text="Enter columns range (C1:C2):", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    def normalize_data():
        dataset = pd.read_csv(dataset_path)
        start, end = map(int, columns_entry.get().split(":"))
        scaler = MinMaxScaler()
        dataset.iloc[:, start:end] = scaler.fit_transform(dataset.iloc[:, start:end])
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_normalize" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"Normalized data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=normalize_data)
    execute_button.pack(pady=10)

################################################### standardize_data ####################################################
def standardize_data_window(window):
    columns_label = Label(window, text="Enter columns range (C1:C2):", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    def standardize_data():
        dataset = pd.read_csv(dataset_path)
        start, end = map(int, columns_entry.get().split(":"))
        scaler = StandardScaler()
        dataset.iloc[:, start:end] = scaler.fit_transform(dataset.iloc[:, start:end])
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_standardize" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"standardized data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=standardize_data)
    execute_button.pack(pady=10)

################################################### encode_categorical_data ####################################################
def encode_categorical_data_window(window):
    columns_label = Label(window, text="Enter columns (C1,C5,C7,C3,...):", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    encoding_type_label = Label(window, text="Select encoding type:", font=("Monaco", 12), bg="#08416B", fg="white")
    encoding_type_label.pack(pady=10)

    encoding_type_var = StringVar(window)
    encoding_type_var.set("OneHotEncoder")
    encoding_type_menu = OptionMenu(window, encoding_type_var, "OneHotEncoder", "LabelEncoder")
    encoding_type_menu.pack(pady=10)

    def encode_categorical_data():
        dataset = pd.read_csv(dataset_path)
        columns = [int(col) for col in columns_entry.get().split(",")]

        if encoding_type_var.get() == "OneHotEncoder":
            ct = ColumnTransformer([('encoder', OneHotEncoder(), columns)], remainder='passthrough')
            dataset = pd.DataFrame(ct.fit_transform(dataset))
            print(dataset.head)
        elif encoding_type_var.get() == "LabelEncoder":
            le = LabelEncoder()
            for col in columns:
                column=dataset.columns.values[col]
                dataset[column] = le.fit_transform(dataset[column])
                print(dataset.head)
        
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_{encoding_type_var.get()}" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"standardized data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=encode_categorical_data)
    execute_button.pack(pady=10)

################################################### feature_selection ####################################################
def feature_selection_window(window):
    columns_label = Label(window, text="Enter columns range (C1:C2):", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    components_label = Label(window, text="Enter number of components:", font=("Monaco", 12), bg="#08416B", fg="white")
    components_label.pack(pady=10)

    components_entry = Entry(window, font=("Monaco", 12))
    components_entry.pack(pady=10)

    def feature_selection():
        dataset = pd.read_csv(dataset_path)
        start, end = map(int, columns_entry.get().split(":"))
        n_components = int(components_entry.get())
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(dataset.iloc[:, start:end])
        pca_columns = [f"Feature_{i+1}" for i in range(n_components)]
        dataset.drop(dataset.columns[start:end], axis=1, inplace=True)
        pca_df = pd.DataFrame(pca_result, columns=pca_columns)
        dataset = pd.concat([pca_df,dataset], axis=1)
        
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_Feature_Selection" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"standardized data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=feature_selection)
    execute_button.pack(pady=10)

################################################### handle_imbalanced_data ####################################################
def handle_imbalanced_data_window(window):
    columns_label = Label(window, text="Enter target column index:", font=("Monaco", 12), bg="#08416B", fg="white")
    columns_label.pack(pady=10)

    columns_entry = Entry(window, font=("Monaco", 12))
    columns_entry.pack(pady=10)

    def handle_imbalanced_data():
        dataset = pd.read_csv(dataset_path)
        target_column_index = int(columns_entry.get())
        target_column = dataset.columns[target_column_index]

        x = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        smote = SMOTE()
        x_res, y_res = smote.fit_resample(x, y)
        balanced_dataset = pd.concat([x_res, y_res], axis=1)
        print(balanced_dataset.head())
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_balance" + extension)
        balanced_dataset.to_csv(new_dataset_path, index=False)
        print(f"Balanced data saved to: {new_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=handle_imbalanced_data)
    execute_button.pack(pady=10)

################################################### split_data ####################################################
def split_data_window(window):
    test_size_label = Label(window, text="Enter test size (0.1 or 0.5 ...etc):", font=("Monaco", 12), bg="#08416B", fg="white")
    test_size_label.pack(pady=10)

    test_size_entry = Entry(window, font=("Monaco", 12))
    test_size_entry.pack(pady=10)

    target_column_label = Label(window, text="Enter target column index:", font=("Monaco", 12), bg="#08416B", fg="white")
    target_column_label.pack(pady=10)

    target_column_entry = Entry(window, font=("Monaco", 12))
    target_column_entry.pack(pady=10)

    def split_data():
        dataset = pd.read_csv(dataset_path)
        test_size = float(test_size_entry.get())
        target_column_index = int(target_column_entry.get())
        target_column = dataset.columns[target_column_index]

        x = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        train_dataset = pd.concat([x_train, y_train], axis=1)
        test_dataset = pd.concat([x_test, y_test], axis=1)

        print(f"Training data \n: {train_dataset.head()}")
        print(f"Test data \n: {test_dataset.head()}")

        filename, extension = os.path.splitext(dataset_path)
        train_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_train" + extension)
        test_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_test" + extension)

        train_dataset.to_csv(train_dataset_path, index=False)
        test_dataset.to_csv(test_dataset_path, index=False)

        print(f"Training data saved to: {train_dataset_path}")
        print(f"Test data saved to: {test_dataset_path}")

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=split_data)
    execute_button.pack(pady=10)

##########################################################################################################################################
############################################################## Classification ############################################################
##########################################################################################################################################

def open_classification_window():
    main_window.withdraw()
    classify_window = Toplevel()
    classify_window.title("Classification")
    classify_window.geometry("400x400")
    classify_window.resizable(False, False)
    classify_window.configure(background="#08416B")
    back_button = Button(classify_window, text="< Back", font=("Monaco", 12, "bold"), bg="#ffd343", width='8', command=lambda: close_window(classify_window))
    back_button.pack()
    back_button.place(x=20, y=360)

    classification_methods = [
        ("SVM", classify_svm_window),
        ("KNN", classify_knn_window),
        ("ANN", classify_ann_window),
        ("Decision Tree", classify_dt_window)
    ]

    row = 0
    for method_name, method_function in classification_methods:
        button = Button(classify_window, text=method_name, font=("Monaco", 15, "bold"), bg="#ffd343", width='31', height='2',command=lambda : open_operation_window(method_function))
        button.grid(row=row, column=0, padx=10, pady=10)
        row += 1

def classify_svm_window(window):
    execute_classification(window, SVC())

def classify_knn_window(window):
    execute_classification(window, KNeighborsClassifier())

def classify_ann_window(window):
    execute_classification(window, MLPClassifier())

def classify_dt_window(window):
    execute_classification(window, DecisionTreeClassifier())

def execute_classification(window, classifier):
    target_column_label = Label(window, text="Enter target column index:", font=("Monaco", 12), bg="#08416B", fg="white")
    target_column_label.pack(pady=10)

    target_column_entry = Entry(window, font=("Monaco", 12))
    target_column_entry.pack(pady=10)

    train_size_label = Label(window, text="Enter train size (default 0.8):", font=("Monaco", 12), bg="#08416B", fg="white")
    train_size_label.pack(pady=10)

    train_size_entry = Entry(window, font=("Monaco", 12))
    train_size_entry.pack(pady=10)

    def run_classification():
        dataset = pd.read_csv(dataset_path)
        target_column_index = int(target_column_entry.get())
        target_column = dataset.columns[target_column_index]

        x = dataset.drop(columns=[target_column])
        y = dataset[target_column]

        train_size = float(train_size_entry.get()) if train_size_entry.get() else 0.8

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_size))

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        classification_dataset = pd.DataFrame(x_test)
        classification_dataset[target_column] = y_pred

        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_{classifier}" + extension)
        classification_dataset.to_csv(new_dataset_path, index=False)
        print(f"Classification data saved to: {new_dataset_path}")

        result_label = Label(window, text=f"Accuracy: {accuracy:.2f}", font=("Monaco", 12), bg="#08416B", fg="white")
        result_label.pack(pady=10)

    execute_button = Button(window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=run_classification)
    execute_button.pack(pady=10)
##########################################################################################################################################
################################################################ Clustering ##############################################################
##########################################################################################################################################
def open_clustering_window():
    main_window.withdraw()
    clustering_window = Toplevel()
    clustering_window.title("Clustering")
    clustering_window.geometry("400x200")
    clustering_window.resizable(False, False)
    clustering_window.configure(background="#08416C")

    n_clusters_label = Label(clustering_window, text="Enter number of clusters:", font=("Monaco", 12), bg="#08416B", fg="white")
    n_clusters_label.pack(pady=10)

    n_clusters_entry = Entry(clustering_window, font=("Monaco", 12))
    n_clusters_entry.pack(pady=10)

    def execute_clustering():
        n_clusters = int(n_clusters_entry.get())
        dataset = pd.read_csv(dataset_path)
        kmeans = KMeans(n_clusters=n_clusters)
        dataset['cluster'] = kmeans.fit_predict(dataset)
        
        filename, extension = os.path.splitext(dataset_path)
        new_dataset_path = os.path.join(os.path.dirname(dataset_path), f"{filename}_after_clustering" + extension)
        dataset.to_csv(new_dataset_path, index=False)
        print(f"Clustered data saved to: {new_dataset_path}")
        print(f"Clustering with {n_clusters} clusters")

    def close_clustering_window():
        clustering_window.destroy()
        main_window.deiconify()

    execute_button = Button(clustering_window, text="Execute", font=("Monaco", 12, "bold"), bg="#ffd343", command=execute_clustering)
    execute_button.pack(pady=10)

    back_button = Button(clustering_window, text="Back", font=("Monaco", 12, "bold"), bg="#ffd343", command=close_clustering_window)
    back_button.pack(pady=10)



main_window.mainloop()
