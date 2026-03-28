from __future__ import annotations
import json
import joblib
import pandas as pd
import streamlit as st
from src.config import BEST_MODEL_FILE, RESULTS_FILE, TARGET_LABELS
from src.data_utils import load_adult_train_test, dataset_profile

st.set_page_config(page_title='Adult Income Predictor Pro', page_icon='💼', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 22px; background: linear-gradient(135deg,#0f172a 0%,#1d4ed8 55%,#2563eb 100%); color:white; box-shadow:0 18px 42px rgba(37,99,235,.22); margin-bottom:1rem;}
</style>
''', unsafe_allow_html=True)

@st.cache_data
def get_data():
    train_df, test_df = load_adult_train_test()
    return train_df, test_df, dataset_profile(train_df, test_df)

@st.cache_resource
def get_bundle():
    model = joblib.load(BEST_MODEL_FILE) if BEST_MODEL_FILE.exists() else None
    results = json.loads(RESULTS_FILE.read_text(encoding='utf-8')) if RESULTS_FILE.exists() else {}
    return model, results

train_df, test_df, profile = get_data()
model, results = get_bundle()
full_df = pd.concat([train_df.assign(split='Train'), test_df.assign(split='Test')], ignore_index=True)

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title('Adult Income Predictor Pro')
    page = st.radio('Navigate', ['Dashboard','Data Explorer','Model Lab','Predict'])
    st.caption('Coursework-ready Streamlit software bundle')

st.markdown("<div class='hero'><h2 style='margin:0;'>Adult / Census Income ML Software</h2><p style='margin:.5rem 0 0 0;'>Dataset exploration, model comparison, and live prediction in one polished app.</p></div>", unsafe_allow_html=True)

if page == 'Dashboard':
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Total rows', f"{profile['rows_total']:,}")
    c2.metric('Train / Test', f"{profile['train_rows']:,} / {profile['test_rows']:,}")
    c3.metric('Features', profile['feature_count'])
    c4.metric('>50K share', f"{profile['positive_rate']*100:.1f}%")
    left,right = st.columns([1.2,1])
    with left:
        st.subheader('Project overview')
        st.write('Use this software in the viva to show the dataset, preprocessing logic, model comparison, and one live prediction. It is intentionally focused and coursework-friendly.')
        if results:
            st.info(f"Deployment model: **{results.get('deployment_model','N/A')}** | Bundle mode: **{results.get('bundle_mode','unknown')}**")
        st.dataframe(full_df.head(12), use_container_width=True)
    with right:
        st.subheader('Missing values')
        missing = profile.get('missing_by_column', {})
        if missing:
            missing_df = pd.DataFrame({'column': list(missing.keys()), 'missing_values': list(missing.values())})
            st.bar_chart(missing_df.set_index('column'))
        st.subheader('Income distribution')
        target_counts = full_df['income'].value_counts().rename_axis('income').reset_index(name='count')
        st.bar_chart(target_counts.set_index('income'))

elif page == 'Data Explorer':
    st.subheader('Explore the data')
    tab1, tab2, tab3 = st.tabs(['Preview','Numeric','Categorical'])
    with tab1:
        split = st.selectbox('Split', ['All','Train','Test'])
        view = full_df if split == 'All' else full_df[full_df['split'] == split]
        st.dataframe(view.head(50), use_container_width=True)
    with tab2:
        num_col = st.selectbox('Numeric feature', ['age','education_num','hours_per_week','capital_gain','capital_loss','fnlwgt'])
        st.bar_chart(full_df.groupby('income')[num_col].mean())
    with tab3:
        cat_col = st.selectbox('Categorical feature', ['education','workclass','marital_status','occupation','relationship','sex','native_country'])
        counts = full_df[cat_col].fillna('Missing').value_counts().head(12).rename_axis(cat_col).reset_index(name='count')
        st.bar_chart(counts.set_index(cat_col))

elif page == 'Model Lab':
    st.subheader('Model comparison')
    if not results:
        st.warning('Run python train_model.py first to generate the bundle.')
    else:
        models_df = pd.DataFrame(results['models']).T.reset_index().rename(columns={'index':'model'})
        cols = [c for c in ['model','accuracy','precision','recall','f1','roc_auc','fit_seconds','note'] if c in models_df.columns]
        st.dataframe(models_df[cols], use_container_width=True)
        st.caption('Packaged note: the saved deployment model is the Random Forest. Re-run training locally for your final report tables if you want everything generated fresh.')
        if 'feature_importance' in results:
            st.subheader('Top feature importance')
            fi_df = pd.DataFrame(results['feature_importance'])
            st.bar_chart(fi_df.set_index('feature'))
        cm = results.get('models',{}).get('Random Forest',{}).get('confusion_matrix')
        if cm:
            st.subheader('Random Forest confusion matrix')
            cm_df = pd.DataFrame(cm, index=['Actual <=50K','Actual >50K'], columns=['Pred <=50K','Pred >50K'])
            st.dataframe(cm_df, use_container_width=True)

else:
    st.subheader('Live prediction')
    if model is None:
        st.error('No saved deployment model found. Run python train_model.py first.')
    else:
        col1,col2,col3 = st.columns(3)
        with col1:
            age = st.slider('Age', 18, 90, 37)
            workclass = st.selectbox('Workclass', sorted(train_df['workclass'].dropna().unique().tolist()))
            education = st.selectbox('Education', sorted(train_df['education'].dropna().unique().tolist()))
            education_num = st.slider('Education num', 1, 16, 10)
            marital_status = st.selectbox('Marital status', sorted(train_df['marital_status'].dropna().unique().tolist()))
        with col2:
            occupation = st.selectbox('Occupation', sorted(train_df['occupation'].dropna().unique().tolist()))
            relationship = st.selectbox('Relationship', sorted(train_df['relationship'].dropna().unique().tolist()))
            race = st.selectbox('Race', sorted(train_df['race'].dropna().unique().tolist()))
            sex = st.selectbox('Sex', sorted(train_df['sex'].dropna().unique().tolist()))
            native_country = st.selectbox('Native country', sorted(train_df['native_country'].dropna().unique().tolist()))
        with col3:
            fnlwgt = st.number_input('fnlwgt', min_value=1000, max_value=1500000, value=189778, step=1000)
            capital_gain = st.number_input('Capital gain', min_value=0, max_value=100000, value=0, step=100)
            capital_loss = st.number_input('Capital loss', min_value=0, max_value=5000, value=0, step=10)
            hours_per_week = st.slider('Hours per week', 1, 99, 40)
        if st.button('Predict income class', type='primary'):
            row = pd.DataFrame([{
                'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education, 'education_num': education_num,
                'marital_status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'sex': sex,
                'capital_gain': capital_gain, 'capital_loss': capital_loss, 'hours_per_week': hours_per_week, 'native_country': native_country,
            }])
            pred = int(model.predict(row)[0])
            conf = float(model.predict_proba(row)[0][pred])
            label = TARGET_LABELS[pred]
            st.success(f'Predicted income class: **{label}**')
            st.progress(min(max(conf, 0.0), 1.0), text=f'Model confidence: {conf:.1%}')
            hist = row.copy(); hist['prediction'] = label; hist['confidence'] = round(conf,4)
            st.session_state.history.append(hist)
        if st.session_state.history:
            hist_df = pd.concat(st.session_state.history, ignore_index=True)
            st.subheader('Prediction history')
            st.dataframe(hist_df, use_container_width=True)
            st.download_button('Download prediction history', data=hist_df.to_csv(index=False).encode('utf-8'), file_name='prediction_history.csv', mime='text/csv')
