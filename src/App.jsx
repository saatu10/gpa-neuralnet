import React, { useState, useEffect, useMemo } from 'react';
import { 
  Calculator, 
  TrendingUp, 
  BookOpen, 
  Brain, 
  Plus, 
  Trash2, 
  Save, 
  RefreshCw,
  Award,
  AlertCircle,
  Terminal,
  Cpu,
  Activity
} from 'lucide-react';

/**
 * MINI NUMPY & SKLEARN IMPLEMENTATION (JavaScript)
 * Student Note: Implemented lightweight OLS and K-Means for client-side inference.
 */

const MiniNumPy = {
  mean: (arr) => arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length,
  
  // Standard Deviation (Previously used for "Consistency Score")
  std: (arr) => {
    if (arr.length === 0) return 0;
    const mu = arr.reduce((a, b) => a + b, 0) / arr.length;
    const sumSqDiff = arr.reduce((sum, val) => sum + Math.pow(val - mu, 2), 0);
    return Math.sqrt(sumSqDiff / arr.length);
  },

  dot: (a, b) => a.map((x, i) => a[i] * b[i]).reduce((m, n) => m + n),
  subtract: (a, b) => a.map((x, i) => x - b[i]),
  multiply: (a, b) => a.map((x, i) => x * b[i]),
  power: (arr, p) => arr.map(x => Math.pow(x, p)),
  sum: (arr) => arr.reduce((a, b) => a + b, 0),
};

class MiniLinearRegression {
  constructor() {
    this.slope = 0;
    this.intercept = 0;
    this.r_squared = 0;
  }

  fit(X, y) {
    const n = X.length;
    if (n === 0) return;

    const x_mean = MiniNumPy.mean(X);
    const y_mean = MiniNumPy.mean(y);

    const numerator = MiniNumPy.sum(
      MiniNumPy.multiply(
        X.map(xi => xi - x_mean),
        y.map(yi => yi - y_mean)
      )
    );

    const denominator = MiniNumPy.sum(
      MiniNumPy.power(X.map(xi => xi - x_mean), 2)
    );

    this.slope = denominator === 0 ? 0 : numerator / denominator;
    this.intercept = y_mean - (this.slope * x_mean);
    
    // Calculate simple R^2 (Coefficient of Determination)
    const y_pred = this.predict(X);
    const ss_res = MiniNumPy.sum(MiniNumPy.power(MiniNumPy.subtract(y, y_pred), 2));
    const ss_tot = MiniNumPy.sum(MiniNumPy.power(y.map(yi => yi - y_mean), 2));
    this.r_squared = ss_tot === 0 ? 1 : 1 - (ss_res / ss_tot);
  }

  predict(X_new) {
    return X_new.map(x => (this.slope * x) + this.intercept);
  }
}

class MiniKMeans {
  constructor(k = 3) {
    this.k = k;
    this.centroids = [];
    this.labels = [];
    this.inertia = 0;
  }

  fit(data) {
    if (data.length < this.k) {
        this.centroids = data;
        this.labels = data.map((_, i) => i);
        return;
    }

    let centroids = data.slice(0, this.k);
    let labels = new Array(data.length).fill(0);
    let iterations = 0;
    const maxIter = 100;

    while (iterations < maxIter) {
      labels = data.map(point => {
        const distances = centroids.map(c => Math.abs(point - c));
        return distances.indexOf(Math.min(...distances));
      });

      const newCentroids = [];
      for (let i = 0; i < this.k; i++) {
        const clusterPoints = data.filter((_, idx) => labels[idx] === i);
        if (clusterPoints.length > 0) {
          newCentroids.push(MiniNumPy.mean(clusterPoints));
        } else {
          newCentroids.push(centroids[i]);
        }
      }

      if (JSON.stringify(newCentroids) === JSON.stringify(centroids)) break;
      centroids = newCentroids;
      iterations++;
    }

    this.centroids = centroids;
    this.labels = labels;
    
    const indexedCentroids = centroids.map((val, idx) => ({val, idx}));
    indexedCentroids.sort((a, b) => a.val - b.val);
    
    const remap = {};
    indexedCentroids.forEach((item, newRank) => {
        remap[item.idx] = newRank;
    });
    
    this.labels = this.labels.map(l => remap[l]);
    this.centroids = indexedCentroids.map(c => c.val);
  }
}

// --- APP CONSTANTS ---

const GRADE_POINTS = {
  'A+': 4.0, 'A': 4.0, 'A-': 3.7,
  'B+': 3.3, 'B': 3.0, 'B-': 2.7,
  'C+': 2.3, 'C': 2.0, 'C-': 1.7,
  'D+': 1.3, 'D': 1.0, 'F': 0.0
};

const getGradeFromMarks = (marks) => {
  if (!marks && marks !== 0) return 'F';
  if (marks >= 97) return 'A+';
  if (marks >= 93) return 'A';
  if (marks >= 90) return 'A-';
  if (marks >= 87) return 'B+';
  if (marks >= 83) return 'B';
  if (marks >= 80) return 'B-';
  if (marks >= 77) return 'C+';
  if (marks >= 73) return 'C';
  if (marks >= 70) return 'C-';
  if (marks >= 67) return 'D+';
  if (marks >= 60) return 'D';
  return 'F';
};

const SAMPLE_DATA = [
  { id: 1, name: "Intro to CS", credits: 4, grade: "A", semester: 1, marks: 95 },
  { id: 2, name: "Calculus I", credits: 4, grade: "B+", semester: 1, marks: 88 },
  { id: 3, name: "Physics I", credits: 3, grade: "B-", semester: 1, marks: 81 },
  { id: 4, name: "Data Structures", credits: 4, grade: "A-", semester: 2, marks: 91 },
  { id: 5, name: "Linear Algebra", credits: 3, grade: "B", semester: 2, marks: 85 },
  { id: 6, name: "Algorithms", credits: 4, grade: "A", semester: 3, marks: 94 },
  { id: 7, name: "Web Dev", credits: 3, grade: "A+", semester: 3, marks: 98 },
  { id: 8, name: "Database Systems", credits: 3, grade: "B+", semester: 3, marks: 89 },
];

// --- COMPONENTS ---

const Card = ({ children, className = "" }) => (
  <div className={`bg-white rounded-lg shadow-sm border border-slate-200 p-6 ${className}`}>
    {children}
  </div>
);

const Badge = ({ children, color = "blue" }) => {
  const colors = {
    blue: "bg-blue-50 text-blue-700 border border-blue-200",
    green: "bg-green-50 text-green-700 border border-green-200",
    yellow: "bg-amber-50 text-amber-700 border border-amber-200",
    red: "bg-red-50 text-red-700 border border-red-200",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${colors[color] || colors.blue}`}>
      {children}
    </span>
  );
};

export default function GPAApp() {
  const [courses, setCourses] = useState(SAMPLE_DATA);
  const [newCourse, setNewCourse] = useState({ name: "", credits: 3, grade: "A", semester: 1, marks: "" });
  const [activeTab, setActiveTab] = useState("dashboard");

  // --- ANALYTICS & CALCULATION ---

  const stats = useMemo(() => {
    // 1. Basic Stats
    const totalCredits = courses.reduce((sum, c) => sum + parseInt(c.credits), 0);
    const totalPoints = courses.reduce((sum, c) => sum + (GRADE_POINTS[c.grade] * parseInt(c.credits)), 0);
    const gpa = totalCredits === 0 ? 0 : (totalPoints / totalCredits).toFixed(2);

    // 2. Semester Trends
    const semesters = [...new Set(courses.map(c => c.semester))].sort((a, b) => a - b);
    const semesterData = semesters.map(sem => {
      const semCourses = courses.filter(c => c.semester === sem);
      const semCredits = semCourses.reduce((sum, c) => sum + parseInt(c.credits), 0);
      const semPoints = semCourses.reduce((sum, c) => sum + (GRADE_POINTS[c.grade] * parseInt(c.credits)), 0);
      return {
        semester: sem,
        gpa: semCredits === 0 ? 0 : semPoints / semCredits
      };
    });

    // Calculate Variance/StdDev for "Consistency"
    const gpaValues = semesterData.map(d => d.gpa);
    const stdDev = MiniNumPy.std(gpaValues).toFixed(3);

    // 3. ML Model: Linear Regression
    let predictedGPA = 0;
    let trend = "stable";
    let nextSem = 1;
    let rSquared = 0;
    let modelParams = { slope: 0, intercept: 0 };

    if (semesterData.length > 1) {
      const model = new MiniLinearRegression();
      const X = semesterData.map(d => d.semester);
      const y = semesterData.map(d => d.gpa);
      
      model.fit(X, y);
      nextSem = Math.max(...X) + 1;
      const prediction = model.predict([nextSem])[0];
      predictedGPA = Math.min(4.0, Math.max(0.0, prediction)).toFixed(2);
      trend = model.slope > 0.05 ? "rising" : model.slope < -0.05 ? "falling" : "stable";
      rSquared = model.r_squared.toFixed(2);
      modelParams = { slope: model.slope.toFixed(3), intercept: model.intercept.toFixed(3) };
    }

    // 4. ML Model: K-Means Clustering
    const numericalGrades = courses.map(c => GRADE_POINTS[c.grade]);
    const kMeans = new MiniKMeans(3);
    kMeans.fit(numericalGrades);
    
    const clusteredCourses = courses.map((c, i) => ({
      ...c,
      cluster: kMeans.labels[i] 
    }));

    const subjectStrengths = {
      strong: clusteredCourses.filter(c => c.cluster === 2),
      average: clusteredCourses.filter(c => c.cluster === 1),
      weak: clusteredCourses.filter(c => c.cluster === 0)
    };

    return { 
      gpa, 
      totalCredits, 
      semesterData, 
      predictedGPA, 
      trend, 
      nextSem,
      subjectStrengths,
      stdDev,
      rSquared,
      modelParams
    };
  }, [courses]);

  // --- HANDLERS ---

  const addCourse = (e) => {
    e.preventDefault();
    const id = courses.length > 0 ? Math.max(...courses.map(c => c.id)) + 1 : 1;
    setCourses([...courses, { ...newCourse, id }]);
    setNewCourse({ name: "", credits: 3, grade: "A", semester: 1, marks: "" });
  };

  const handleMarksChange = (e) => {
    const val = e.target.value;
    const numVal = parseInt(val);
    
    if (val === "") {
        setNewCourse({ ...newCourse, marks: "" });
        return;
    }

    if (!isNaN(numVal) && numVal >= 0 && numVal <= 100) {
        setNewCourse({ 
            ...newCourse, 
            marks: numVal, 
            grade: getGradeFromMarks(numVal) 
        });
    }
  };

  const deleteCourse = (id) => {
    setCourses(courses.filter(c => c.id !== id));
  };

  const loadSample = () => {
    setCourses(SAMPLE_DATA);
  };

  // --- RENDER HELPERS ---

  const renderTrendIcon = () => {
    if (stats.trend === "rising") return <TrendingUp className="text-green-500 w-5 h-5" />;
    if (stats.trend === "falling") return <TrendingUp className="text-red-500 w-5 h-5 transform rotate-180" />;
    return <Activity className="text-slate-400 w-5 h-5" />;
  };

  return (
    <div className="min-h-screen bg-slate-100 font-sans text-slate-800">
      {/* HEADER */}
      <header className="bg-slate-900 text-white border-b border-slate-800 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-1.5 bg-green-500/10 rounded-md border border-green-500/30">
                <Terminal className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <h1 className="text-lg font-mono font-bold tracking-tight text-slate-200">GPA_Predictor<span className="text-green-500">.py</span></h1>
              <div className="flex items-center space-x-2 text-[10px] text-slate-400 font-mono">
                <span>v3.0.1-beta</span>
                <span>•</span>
                <span>local_env</span>
              </div>
            </div>
          </div>
          <nav className="flex space-x-1 bg-slate-800 p-1 rounded-md border border-slate-700">
            <button 
              onClick={() => setActiveTab("dashboard")}
              className={`px-3 py-1.5 rounded text-xs font-mono transition-all ${activeTab === "dashboard" ? "bg-slate-700 text-green-400 shadow-sm" : "text-slate-400 hover:text-slate-200"}`}
            >
              ./dashboard
            </button>
            <button 
              onClick={() => setActiveTab("entry")}
              className={`px-3 py-1.5 rounded text-xs font-mono transition-all ${activeTab === "entry" ? "bg-slate-700 text-green-400 shadow-sm" : "text-slate-400 hover:text-slate-200"}`}
            >
              ./data_entry
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        
        {activeTab === "dashboard" && (
          <div className="space-y-6">
            
            {/* TOP STATS ROW */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* MAIN GPA - Styled like a metric card */}
              <Card className="border-t-4 border-t-indigo-500">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-slate-500 font-mono text-xs uppercase">Cumulative_GPA</h3>
                  <Award className="w-5 h-5 text-indigo-500" />
                </div>
                <div className="flex items-baseline space-x-2">
                  <div className="text-4xl font-mono font-bold text-slate-800">{stats.gpa}</div>
                  <span className="text-xs text-slate-400 font-mono">/ 4.00</span>
                </div>
                <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-xs font-mono text-slate-500">
                    <span>Credits: {stats.totalCredits}</span>
                    <span className="bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded">n={courses.length}</span>
                </div>
              </Card>

              {/* PREDICTION CARD - Technical view */}
              <Card className="border-t-4 border-t-purple-500">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-slate-500 font-mono text-xs uppercase">Linear_Model.predict()</h3>
                  {renderTrendIcon()}
                </div>
                <div className="flex items-baseline space-x-2">
                  <span className="text-4xl font-mono font-bold text-slate-800">{stats.predictedGPA}</span>
                  <span className="text-xs text-purple-600 bg-purple-50 px-1 rounded font-mono">Sem {stats.nextSem}</span>
                </div>
                <div className="mt-2 pt-2 border-t border-slate-100 text-[10px] font-mono text-slate-500 space-y-0.5">
                    <div className="flex justify-between">
                        <span>Slope (m): {stats.modelParams.slope}</span>
                        <span>R²: {stats.rSquared}</span>
                    </div>
                </div>
              </Card>

              {/* CLUSTER CARD - Technical View */}
              <Card className="border-t-4 border-t-teal-500">
                <h3 className="text-slate-500 font-mono text-xs uppercase mb-3">KMeans(k=3).fit_predict()</h3>
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex items-center justify-between p-1 bg-green-50 rounded border border-green-100">
                    <span className="text-green-700">Cluster_2 (High)</span>
                    <span className="font-bold text-green-800">{stats.subjectStrengths.strong.length} samples</span>
                  </div>
                  <div className="flex items-center justify-between p-1 bg-amber-50 rounded border border-amber-100">
                    <span className="text-amber-700">Cluster_1 (Avg)</span>
                    <span className="font-bold text-amber-800">{stats.subjectStrengths.average.length} samples</span>
                  </div>
                  <div className="flex items-center justify-between p-1 bg-red-50 rounded border border-red-100">
                    <span className="text-red-700">Cluster_0 (Low)</span>
                    <span className="font-bold text-red-800">{stats.subjectStrengths.weak.length} samples</span>
                  </div>
                </div>
              </Card>
            </div>

            {/* CHART & CONSOLE ROW */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* CHART */}
              <Card className="lg:col-span-2">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="font-mono text-sm font-bold flex items-center text-slate-700">
                    <Activity className="w-4 h-4 mr-2 text-slate-400" />
                    Performance_Trajectory
                    </h3>
                    <div className="flex items-center space-x-2 text-[10px] font-mono text-slate-400">
                        <span className="w-2 h-2 bg-indigo-500 rounded-sm"></span>
                        <span>Actual</span>
                        <span className="w-2 h-2 bg-slate-300 rounded-sm ml-2"></span>
                        <span>Reg Line</span>
                    </div>
                </div>

                <div className="h-64 flex items-end justify-between px-4 pb-2 space-x-4 relative border-l border-b border-slate-300">
                  {/* Grid lines - Graph paper style */}
                  <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20">
                    <div className="border-b border-slate-400 border-dashed w-full h-0"></div>
                    <div className="border-b border-slate-400 border-dashed w-full h-0"></div>
                    <div className="border-b border-slate-400 border-dashed w-full h-0"></div>
                    <div className="border-b border-slate-400 border-dashed w-full h-0"></div>
                    <div className="border-b border-slate-400 border-dashed w-full h-0"></div>
                  </div>

                  {stats.semesterData.length > 0 ? stats.semesterData.map((d, i) => (
                    <div key={i} className="flex flex-col items-center group relative w-full z-10">
                      <div 
                        className="w-full max-w-[30px] bg-indigo-500 hover:bg-indigo-600 transition-all relative"
                        style={{ height: `${(d.gpa / 4) * 100}%` }}
                      >
                         <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-slate-900 text-green-400 text-[10px] font-mono py-1 px-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-20">
                           val: {d.gpa.toFixed(2)}
                         </div>
                      </div>
                      <span className="mt-3 text-[10px] font-mono text-slate-500">S{d.semester}</span>
                    </div>
                  )) : (
                    <div className="w-full h-full flex items-center justify-center font-mono text-xs text-slate-400">
                      [waiting for input data...]
                    </div>
                  )}
                </div>
              </Card>
              
              {/* TERMINAL / LOGS (New "Insights" section) */}
              <div className="bg-slate-900 rounded-lg p-4 font-mono text-xs text-slate-300 shadow-lg flex flex-col h-full border border-slate-700">
                 <div className="flex items-center justify-between mb-3 border-b border-slate-700 pb-2">
                    <span className="flex items-center text-green-500 font-bold">
                        <Terminal className="w-3 h-3 mr-2" />
                        runtime_logs
                    </span>
                    <div className="flex space-x-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-slate-700"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-slate-700"></div>
                    </div>
                 </div>
                 
                 <div className="space-y-2 overflow-y-auto flex-grow custom-scrollbar">
                    <div className="text-slate-500">
                        <span className="text-blue-400">INFO </span> 
                        [System] Initializing MiniNumPy engine...
                    </div>
                    
                    {stats.semesterData.length < 2 ? (
                        <div className="text-yellow-500">
                            <span className="text-yellow-400">WARN </span>
                            Insufficient data points (n={stats.semesterData.length}). Need n&gt;1 for regression.
                        </div>
                    ) : (
                        <>
                            <div>
                                <span className="text-blue-400">INFO </span>
                                Loaded {courses.length} samples.
                            </div>
                            <div>
                                <span className="text-blue-400">INFO </span>
                                Calculating std_dev... <span className="text-green-400">{stats.stdDev}</span>
                            </div>
                            <div>
                                <span className="text-purple-400">EXEC </span>
                                <span className="text-slate-400">MiniLinearRegression.fit(X, y)</span>
                            </div>
                            <div className="pl-4 border-l border-slate-700 text-slate-400">
                                <div>&gt; Epochs: 1 (OLS Closed Form)</div>
                                <div>&gt; Loss (MSE): {(1 - stats.rSquared).toFixed(4)}</div>
                                <div>&gt; Converged.</div>
                            </div>
                            <div>
                                <span className="text-green-400">SUCCESS </span>
                                Predicted Next Term: <span className="font-bold text-white">{stats.predictedGPA}</span>
                            </div>
                        </>
                    )}

                    <div className="pt-2 mt-2 border-t border-slate-800">
                        <span className="text-teal-400">ANALYSIS </span>
                        Top Feature:
                        <br/>
                        <span className="text-white">
                        {stats.subjectStrengths.strong.length > 0 
                            ? `"${stats.subjectStrengths.strong[0].name}" (Cluster_2)`
                            : "None"}
                        </span>
                    </div>
                    
                    <div className="mt-2">
                        <span className="text-slate-500 animate-pulse">_</span>
                    </div>
                 </div>
              </div>

            </div>
          </div>
        )}

        {activeTab === "entry" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* INPUT FORM */}
            <div className="lg:col-span-1">
              <Card>
                <h2 className="text-sm font-bold font-mono mb-4 flex items-center text-slate-700">
                  <Plus className="w-4 h-4 mr-2" /> append_course()
                </h2>
                <form onSubmit={addCourse} className="space-y-4">
                  <div>
                    <label className="block text-xs font-mono font-medium text-slate-500 mb-1">course_name (str)</label>
                    <input 
                      type="text" 
                      required
                      value={newCourse.name}
                      onChange={e => setNewCourse({...newCourse, name: e.target.value})}
                      className="w-full px-3 py-2 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-green-500 focus:outline-none font-mono"
                      placeholder="'Neural Networks'"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-mono font-medium text-slate-500 mb-1">marks (int)</label>
                        <input 
                          type="number" 
                          min="0" max="100"
                          value={newCourse.marks}
                          onChange={handleMarksChange}
                          placeholder="None"
                          className="w-full px-3 py-2 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-green-500 focus:outline-none font-mono"
                        />
                    </div>
                    <div>
                      <label className="block text-xs font-mono font-medium text-slate-500 mb-1">grade (cat)</label>
                      <select 
                        value={newCourse.grade}
                        onChange={e => setNewCourse({...newCourse, grade: e.target.value})}
                        className="w-full px-3 py-2 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-green-500 focus:outline-none font-mono bg-white"
                      >
                        {Object.keys(GRADE_POINTS).map(g => (
                          <option key={g} value={g}>{g}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-mono font-medium text-slate-500 mb-1">credits (int)</label>
                      <input 
                        type="number" 
                        min="0" max="10"
                        value={newCourse.credits}
                        onChange={e => setNewCourse({...newCourse, credits: parseInt(e.target.value) || 0})}
                        className="w-full px-3 py-2 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-green-500 focus:outline-none font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-mono font-medium text-slate-500 mb-1">sem_id (int)</label>
                      <input 
                        type="number" 
                        min="1" max="12"
                        value={newCourse.semester}
                        onChange={e => setNewCourse({...newCourse, semester: parseInt(e.target.value) || 1})}
                        className="w-full px-3 py-2 text-sm border border-slate-300 rounded focus:ring-2 focus:ring-green-500 focus:outline-none font-mono"
                      />
                    </div>
                  </div>

                  <button 
                    type="submit" 
                    className="w-full bg-slate-800 text-green-400 border border-slate-700 py-2 rounded font-mono text-sm hover:bg-slate-700 transition-colors flex items-center justify-center shadow-sm"
                  >
                    <Save className="w-3 h-3 mr-2" /> commit
                  </button>
                  
                  <button 
                    type="button" 
                    onClick={loadSample}
                    className="w-full bg-white text-slate-500 border border-slate-200 py-2 rounded font-mono text-sm hover:bg-slate-50 transition-colors flex items-center justify-center mt-2"
                  >
                    <RefreshCw className="w-3 h-3 mr-2" /> load_sample_dataset
                  </button>
                </form>
              </Card>
            </div>

            {/* TABLE */}
            <div className="lg:col-span-2">
              <Card className="overflow-hidden">
                <div className="flex items-center justify-between mb-4">
                   <h2 className="text-sm font-bold font-mono text-slate-700">df.head({courses.length})</h2>
                   <div className="flex space-x-2">
                     <span className="w-3 h-3 rounded-full bg-red-400"></span>
                     <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                     <span className="w-3 h-3 rounded-full bg-green-400"></span>
                   </div>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse font-mono text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-400 bg-slate-50">
                        <th className="py-2 px-3 font-medium">sem</th>
                        <th className="py-2 px-3 font-medium">course</th>
                        <th className="py-2 px-3 font-medium">cred</th>
                        <th className="py-2 px-3 font-medium">marks</th>
                        <th className="py-2 px-3 font-medium">grade</th>
                        <th className="py-2 px-3 font-medium">pts</th>
                        <th className="py-2 px-3 font-medium text-right">del</th>
                      </tr>
                    </thead>
                    <tbody>
                      {courses.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="py-8 text-center text-slate-400">
                            Dataset is empty.
                          </td>
                        </tr>
                      ) : (
                        courses.sort((a,b) => b.semester - a.semester || b.id - a.id).map(c => (
                          <tr key={c.id} className="border-b border-slate-100 hover:bg-slate-50 group">
                            <td className="py-2 px-3 text-slate-500">{c.semester}</td>
                            <td className="py-2 px-3 font-medium text-slate-700">{c.name}</td>
                            <td className="py-2 px-3 text-slate-500">{c.credits}</td>
                            <td className="py-2 px-3 text-slate-500">
                                {c.marks !== undefined && c.marks !== "" ? c.marks : "NaN"}
                            </td>
                            <td className="py-2 px-3">
                              <Badge color={
                                c.grade.startsWith('A') ? 'green' : 
                                c.grade.startsWith('B') ? 'blue' : 
                                c.grade.startsWith('C') ? 'yellow' : 'red'
                              }>
                                {c.grade}
                              </Badge>
                            </td>
                            <td className="py-2 px-3 text-slate-500">
                              {(GRADE_POINTS[c.grade] * c.credits).toFixed(1)}
                            </td>
                            <td className="py-2 px-3 text-right">
                              <button 
                                onClick={() => deleteCourse(c.id)}
                                className="text-slate-300 hover:text-red-500 transition-colors"
                              >
                                <Trash2 className="w-3 h-3" />
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}