from flask import Flask,request,render_template
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_attrition():
    Age = request.form.get('Age')
    BusinessTravel = request.form.get('BusinessTravel')
    DailyRate = request.form.get('DailyRate')
    Department = request.form.get('Department')
    DistanceFromHome = request.form.get('DistanceFromHome')
    Education = request.form.get('Education')
    EducationField = request.form.get('EducationField')
    EnvironmentSatisfaction = request.form.get('EnvironmentSatisfaction')
    Gender = request.form.get('Gender')
    HourlyRate = request.form.get('HourlyRate')
    JobInvolvement = request.form.get('JobInvolvement')
    JobLevel = request.form.get('JobLevel')
    JobRole = request.form.get('JobRole')
    JobSatisfaction = request.form.get('JobSatisfaction')
    MaritalStatus = request.form.get('MaritalStatus')
    MonthlyIncome = request.form.get('MonthlyIncome')
    MonthlyRate = request.form.get('MonthlyRate')
    NumCompaniesWorked = request.form.get('NumCompaniesWorked')
    OverTime = request.form.get('OverTime')
    PercentSalaryHike = request.form.get('PercentSalaryHike')
    PerformanceRating = request.form.get('PerformanceRating')
    RelationshipSatisfaction = request.form.get('RelationshipSatisfaction')
    StockOptionLevel = request.form.get('StockOptionLevel')
    TotalWorkingYears = request.form.get('TotalWorkingYears')
    TrainingTimesLastYear = request.form.get('TrainingTimesLastYear')
    WorkLifeBalance = request.form.get('WorkLifeBalance')
    YearsAtCompany = request.form.get('YearsAtCompany')
    YearsInCurrentRole = request.form.get('YearsInCurrentRole')
    YearsSinceLastPromotion = request.form.get('YearsSinceLastPromotion')
    YearsWithCurrManager = request.form.get('YearsWithCurrManager')

    #prediction
    test= [Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
       Education, EducationField, EnvironmentSatisfaction, Gender,
       HourlyRate, JobInvolvement, JobLevel, JobRole,
       JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
       NumCompaniesWorked, OverTime, PercentSalaryHike,
       PerformanceRating, RelationshipSatisfaction, StockOptionLevel,
       TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
       YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
       YearsWithCurrManager]
    test = np.array([test]).reshape(1,30)


    result= model.predict(test)
    
    if result[0]==1:
        result= 'Will Leave'
    else:
        result='Will Stay'

    return render_template('index.html',result=result)



if __name__=='__main__':
    app.run(debug=True)

#21.0,2,337.0,2,7.000000,0,2,1,1,1,2,0,8,1,2,187,94,1,0,2,0,1,0,1,3,2,1,0,1,0