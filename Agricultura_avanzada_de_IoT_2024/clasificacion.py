# División de Datos
from sklearn.model_selection import train_test_split

def dividir_datos(data):
    # Separar características y etiqueta
    X = data.drop(columns=['Random', 'Class'])
    y = data['Class']

    # Dividir el conjunto de datos en entrenamiento y prueba (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


# ENTRENAMIENTO DE MODELOS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def entrenar_clasificadores(X_train, y_train):
    # Crear y entrenar los modelos
    log_reg = LogisticRegression(max_iter=1000)
    rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC()

    log_reg.fit(X_train, y_train)
    rand_forest.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    
    return log_reg, rand_forest, svm

# EVALUACION DE MODELOS
from sklearn.metrics import accuracy_score, classification_report

def evaluar_clasificadores(modelos, X_test, y_test):
    log_reg, rand_forest, svm = modelos

    # Realizar predicciones en el conjunto de prueba
    log_reg_pred = log_reg.predict(X_test)
    rand_forest_pred = rand_forest.predict(X_test)
    svm_pred = svm.predict(X_test)

    # Evaluar el rendimiento de los modelos
    log_reg_acc = accuracy_score(y_test, log_reg_pred)
    rand_forest_acc = accuracy_score(y_test, rand_forest_pred)
    svm_acc = accuracy_score(y_test, svm_pred)

    log_reg_report = classification_report(y_test, log_reg_pred)
    rand_forest_report = classification_report(y_test, rand_forest_pred)
    svm_report = classification_report(y_test, svm_pred)

    # Mostrar los resultados
    print("Exactitud de Regresión Logística: ", log_reg_acc)
    print("Exactitud de Random Forest: ", rand_forest_acc)
    print("Exactitud de SVM: ", svm_acc)

    print("\nReporte de Clasificación - Regresión Logística:\n", log_reg_report)
    print("\nReporte de Clasificación - Random Forest:\n", rand_forest_report)
    print("\nReporte de Clasificación - SVM:\n", svm_report)

    return log_reg_pred, rand_forest_pred, svm_pred
