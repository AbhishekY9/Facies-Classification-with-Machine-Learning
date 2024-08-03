import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from function import label_facies ,accuracy, accuracy_adjacent, make_facies_log_plot, compare_facies_plot1, compare_facies_plot2
from model_builder import facies_colors,model_s,model_knn,adjacent_facies,model_logi,model_svm,facies_names_abbreviation
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    st.title('Facies Classification')
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['Well Name'] = data['Well Name'].astype('category')
        data['Formation'] = data['Formation'].astype('category')

        well_names = data['Well Name'].unique()
        selected_well = st.selectbox("Select Well Name", well_names)

        if selected_well:
            data = data[data['Well Name'] == selected_well]

            # Check if 'Facies' column is in the uploaded file
            if 'Facies' in data.columns:


                data_prediction = data.copy()
                well_features = data_prediction[['Facies','GR','ILD_log10','PHIND','NM_M']]
                # print(well_features)
                scalar = StandardScaler()
                X_data_scaled = scalar.fit_transform(well_features)
                # X_data_scaled = scaler.transform(well_features)
                # print(X_data_scaled.shape)

                # Make predictions using randomforest
                data_prediction["Synthesis_PE"] = model_s.predict(X_data_scaled)
                data_prediction["PE"] = np.where(data["PE"].isnull(), data_prediction["Synthesis_PE"], data_prediction["PE"])
                # print(data_prediction)
                final_processed_data = data_prediction.drop("Synthesis_PE", axis=1)
                # print(final_processed_data)

                #feature selection after preprosessing
                y_data = final_processed_data['Facies'].values
                # print(final_well_features.shape)
                final_well_features = final_processed_data[
                    ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']]
                # print(final_well_features.shape)
                scalar1 = StandardScaler()
                X_final_scaled = scalar1.fit_transform(final_well_features)
                # print(X_final_scaled.shape)

                selected_algo = st.selectbox("Choose an Algorithm", ['KNN',"SVM","Logistic Regression"])
                if selected_algo=="KNN":
                    # KNN model buildup
                    model = model_knn
                elif selected_algo=="SVM":
                    # SVM model buildup
                    model = model_svm
                elif selected_algo=="Logistic Regression":
                    # Logistic Regression model
                    model= model_logi

                # print(X_final_scaled[0,:])
                y_pred = model.predict(X_final_scaled)
                final_processed_data['Prediction'] = y_pred
                cv_conf = confusion_matrix(y_data, y_pred)

                # Select plot type
                plot_type = st.selectbox(
                    "Select Plot Type",
                    ["1. Facies Log Plot", "2. Compare Facies Plot", "3. Facies Plot if output not known"]
                )
                if plot_type == "1. Facies Log Plot":
                    fig_facies, ax_facies = make_facies_log_plot(
                        final_processed_data[final_processed_data['Well Name'] == selected_well], facies_colors
                    )
                    st.pyplot(fig_facies)  # Pass the figure object to st.pyplot

                elif plot_type == "2. Compare Facies Plot":
                    # Using a default or fixed column name for comparison
                    comparison_column = 'Prediction'  # Change this to the column you want to compare against
                    fig_compare, ax_compare = compare_facies_plot1(
                        final_processed_data[final_processed_data['Well Name'] == selected_well],
                        comparison_column, facies_colors
                    )
                    st.pyplot(fig_compare)  # Pass the figure object to st.pyplot

                    if y_data is not None:
                        st.write('Optimized facies classification accuracy = %.2f' % accuracy(cv_conf))
                        st.write('Optimized adjacent facies classification accuracy = %.2f' % accuracy_adjacent(cv_conf, adjacent_facies))

                elif plot_type == "3. Facies Plot if output not known":
                    # Using a default or fixed column name for comparison
                    comparison_column = 'Prediction'  # Change this to the column you want to compare against
                    fig_compare, ax_compare = compare_facies_plot2(
                        final_processed_data[final_processed_data['Well Name'] == selected_well],
                        comparison_column, facies_colors
                    )
                    st.pyplot(fig_compare)  # Pass the figure object to st.pyplot
                st.write('')
                st.write('')
                st.write('')
                st.markdown("<p style='font-size:11.5px;'>SS: Nonmarine sandstone, CSiS: Nonmarine coarse siltstone, FSiS: Nonmarine fine siltstone, SiSh: Marine siltstone and shale, MS: Mudstone (limestone), WS: Wackestone (limestone), D: Dolomite, PS: Packstone-grainstone (limestone), BS: Phylloid-algal bafflestone (limestone)</p>", unsafe_allow_html=True)

            else:

                x = data[['GR', 'ILD_log10','PHIND','NM_M']]
                # print(x)
                y = data.PE
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(x)
                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.20, random_state=42)


                model_r = RandomForestRegressor(
                    random_state=6,
                    criterion='absolute_error',
                    max_depth=18,
                    max_features='log2',
                    min_samples_leaf=1,
                    min_samples_split=2,
                    n_estimators=200
                )
                model_r.fit(x_train, y_train)

                well_features = data[['GR', 'ILD_log10', 'PHIND', 'NM_M']]
                scaler = preprocessing.StandardScaler().fit(well_features)
                X_data_scaled = scaler.transform(well_features)
                # print(X_data_scaled)

                # Make predictions
                data["Synthesis_PE"] = model_r.predict(X_data_scaled)
                data["PE"] = np.where(data["PE"].isnull(), data["Synthesis_PE"], data["PE"])
                final_processed_data = data.drop("Synthesis_PE", axis=1)

                final_well_features = final_processed_data[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']]
                scaler = preprocessing.StandardScaler().fit(final_well_features)
                X_final_processed_data = scaler.transform(final_well_features)

                selected_algo = st.selectbox("Choose an Algorithm", ['KNN',"SVM","Logistic Regression"])
                if selected_algo == "KNN":
                    # KNN model buildup
                    model = model_knn
                elif selected_algo == "SVM":
                    # SVM model buildup
                    model = model_svm
                elif selected_algo == "Logistic Regression":
                    # Logistic Regression model
                    model = model_logi

                y_pred = model.predict(X_final_processed_data)
                final_processed_data['Prediction'] = y_pred
                # Select plot type
                plot_type = st.selectbox(
                    "Select Plot Type",
                    ["Predicted Facies Plot if output not known"]
                )
                if plot_type == "Predicted Facies Plot if output not known":
                    # Using a default or fixed column name for comparison
                    comparison_column = 'Prediction'  # Change this to the column you want to compare against
                    fig_compare, ax_compare = compare_facies_plot2(
                        final_processed_data[final_processed_data['Well Name'] == selected_well],
                        comparison_column, facies_colors
                    )
                    st.pyplot(fig_compare)  # Pass the figure object to st.pyplot

                st.write('')
                st.write('')
                st.write('')
                st.markdown(
                    "<p style='font-size:11.5px;'>SS: Nonmarine sandstone, CSiS: Nonmarine coarse siltstone, FSiS: Nonmarine fine siltstone, SiSh: Marine siltstone and shale, MS: Mudstone (limestone), WS: Wackestone (limestone), D: Dolomite, PS: Packstone-grainstone (limestone), BS: Phylloid-algal bafflestone (limestone)</p>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
