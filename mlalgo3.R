# Sur la base de données "cells" du package "modeldata", prédire la variable "class" à partir des autres
# Essayez différents algorithmes en partant des plus simples

### Chargement des librairies

library(tidymodels)
library(datasets)
library(mlbench)
library(forestmodel)
library(rpart)
library(rpart.plot)
library(pROC)
library(naniar)
library(forcats)
library(modeldata)
library(tidyverse)
library(naniar)
library(caret)

### Importation des données
dataC = read.csv("C:\\Users\\pinto\\Downloads\\cells.csv", header = TRUE, sep = ";")
summary(cells) # stats du dataset
dim(dataC) # dimension du fichier

# Afficher le type de chaque variable dans le dataset
str(dataC)
sapply(dataC, class)

# Enlever la première colonne 
dataC <- cells[,-1]

### Nettoyage/Préparation des données

# Vérification de la présence de NA dans les données
sum(is.na(dataC))  # Nombre total de valeurs manquantes
colSums(is.na(dataC))  # Nombre de valeurs manquantes par colonne
vis_miss(dataC)  # Visualisation des valeurs manquantes


# Convertir les variables catégorielles en facteurs (si elles existent)
dataC$class <- as.factor(dataC$class)

# Vérification
str(dataC$class)
table(dataC$class)  # Pour voir la répartition des classes

# Remplacer les virgules par des points dans toutes les colonnes sauf 'class'
dataC[,-1] <- lapply(dataC[,-1], function(x) as.numeric(gsub(",", ".", x)))

# Vérifier que les colonnes sont bien numériques
str(dataC)

### Test des différents algorithmes

# Séparation des données en train (80%) et test (20%)
set.seed(42)  # Une graine pour la reproductibilité
index <- createDataPartition(dataC$class, p = 0.8, list = FALSE)
train_data <- dataC[index, ]
test_data <- dataC[-index, ]

# Régression logistique
reglog <- glm(class ~ ., data = train_data, family = binomial(link="logit"))
probs_log <- predict(reglog, test_data, type = "response")
pred_log <- ifelse(probs_log > 0.5, "PS", "WS")
pred_log <- factor(pred_log, levels = levels(test_data$class))

# Validation croisée pour la régression logistique
ctrl <- trainControl(method = "cv", number = 10)  # Validation croisée en 10 groupes
reglog_cv <- train(class ~ ., data = train_data, method = "glm", family = "binomial", trControl = ctrl)
print(reglog_cv)

# Arbre de décision
tree_model <- rpart(class ~ ., data = train_data, method = "class")
pred_tree <- predict(tree_model, test_data, type = "class")

# Random forest
library(randomForest)
rf_model <- randomForest(class ~ ., data = train_data, ntree = 100)
pred_rf <- predict(rf_model, test_data)

# Forcer les niveaux des prédictions à être les mêmes que ceux des données de test
pred_log <- factor(pred_log, levels = levels(test_data$class))


# Matrices de confusion pour évaluer les modèles
conf_matrix_log <- confusionMatrix(pred_log, test_data$class)
conf_matrix_tree <- confusionMatrix(pred_tree, test_data$class)
conf_matrix_rf <- confusionMatrix(pred_rf, test_data$class)

# Afficher des précisions
print(paste("Précision - Régression logistique :", round(conf_matrix_log$overall["Accuracy"] * 100, 2), "%"))
print(paste("Précision - Arbre de décision :", round(conf_matrix_tree$overall["Accuracy"] * 100, 2), "%"))
print(paste("Précision - Random forest :", round(conf_matrix_rf$overall["Accuracy"] * 100, 2), "%"))

# Afficher les courbes ROC

# Calcul de la courbe ROC
roc_log <- roc(test_data$class, probs_log)
# Afficher la courbe ROC
plot(roc_log, main = "Courbe ROC - Régression Logistique")


# Calcul des probabilités pour l'arbre de décision
probs_tree <- predict(tree_model, test_data, type = "prob")[,2]  # Probabilité pour la classe "PS" (ou positive)
# Calcul de la courbe ROC pour l'arbre de décision
roc_tree <- roc(test_data$class, probs_tree)
# Afficher la courbe ROC pour l'arbre de décision
plot(roc_tree, main = "Courbe ROC - Arbre de Décision")


# Calcul des probabilités pour le modèle Random Forest
probs_rf <- predict(rf_model, test_data, type = "prob")[,2]  # Probabilité pour la classe "PS" (ou positive)
# Calcul de la courbe ROC pour le Random Forest
roc_rf <- roc(test_data$class, probs_rf)
# Afficher la courbe ROC pour le Random Forest
plot(roc_rf, main = "Courbe ROC - Random Forest")



# Préparation de la recette pour la régression logistique
lr_recipe <- recipe(class ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# Modèle de régression logistique avec hyperparamètres à ajuster
lr_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Création du workflow
lr_workflow <-
  workflow() %>% 
  add_recipe(lr_recipe) %>% 
  add_model(lr_mod)

# Validation croisée
dataCV <- vfold_cv(train_data, v = 10, strata = class)

# Définir les métriques pour l'évaluation
model_metrics <- metric_set(accuracy, roc_auc)

# Recherche d'hyperparamètres pour la régression logistique
elastic_grid <- grid_regular(penalty(), mixture(), levels = 25)

# Recherche d'hyperparamètres avec la validation croisée et les métriques définies
tune_lr_res <- lr_workflow %>%
  tune_grid(resamples = dataCV, grid = elastic_grid, control = control_grid(), metrics = model_metrics)

# Afficher les résultats de l'optimisation des hyperparamètres
print(tune_lr_res)

# Entraînement du modèle de régression logistique avec le workflow
lr_fit <- lr_workflow %>%
  fit(data = train_data)

# Extraction des coefficients pour la régression logistique
get_lm_coefs <- function(x) {
  x %>%
    extract_fit_engine() %>%
    tidy()
}

# Extraire et afficher les coefficients
lr_coefs <- get_lm_coefs(lr_fit)
print(lr_coefs)

# Extraction de l'importance des variables pour un modèle d'arbre (comme Random Forest)
get_tree_imp <- function(x) {
  x %>%
    extract_fit_engine() %>%
    vip::vi()
}

rf_imp <- get_tree_imp(rf_fit)
print(rf_imp)

# Analyse en composantes principales (PCA):
library(FactoMineR)
library(factoextra)

res.pca <- PCA(dataC[,-1], scale = TRUE)

# Visualisation de la variance expliquée
fviz_eig(res.pca)

# Visualisation des individus colorés par la classe
fviz_pca_ind(res.pca, habillage = dataC$class)




###Commentaire:

#Les trois modèles testés présentent des performances differentes.

#Pour la précision: 

#- Régression Logistique (17.12%) : La valeur ici est basse et cela pourrait indiquer que celle ci n'arrive pas a bien prédire la variable cible.
#Cela peut etre du à un desequilibre des classes, a des problèmes dans l'étape de préparation des données ou encore le choix d'un mauvais seuil.
#On pourrait ameliorer cela en reajustant le seuil en équilibrant les classes ou encore en vérifiant la multicolinéarité des variables.

#- Arbre de Décision (80.4%) : La valeur ici semble correcte et cela indiquerait que ce modèle serait capable de séparer les differentes classes
#en suivant des règles de décision simples.
#Des problemes tels le sur-entrainement ou sur-ajustement dans le cas ou on aurait trop de variables
#On pourrait ameliorer cela en essayant de regler les hyperparametres ou on essayant une validation croisée pour vérifier la stabilité des performances

#- Random Forest (83.37%): On a obtenu la valeur la plus importante ici. Ceci indiquerait que ce modèle est le plus performant
#Ce modele regroupe plusieurs arbres de décision, ce qui le rend plus robuste aux fluctuations des données et réduit le risque de sur-ajustement par rapport à un arbre de décision simple.
#On pourrait toujours l'ameliorer en faisant du bagging ou en selectionnant les variables importantes et en "excluant" les variables non informatives pour reduire le bruit.


#[Ce fichier sera amelioré ultérieurement]