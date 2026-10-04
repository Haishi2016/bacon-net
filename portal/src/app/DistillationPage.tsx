"use client";

import { useState } from "react";

type DistillJob = {
  id: string;
  teacher: string;
  student: string;
  disease: string;
  compression: string;
  teacherAuc: number;
  studentAuc: number;
  status: "completed" | "running" | "queued";
  progress: number;
};

const JOBS: DistillJob[] = [
  {
    id: "d1",
    teacher: "Breast Cancer · v1.3 ensemble",
    student: "Breast Cancer · edge-lite",
    disease: "Breast Cancer",
    compression: "12.4× smaller",
    teacherAuc: 0.985,
    studentAuc: 0.979,
    status: "completed",
    progress: 100
  },
  {
    id: "d2",
    teacher: "Lung Nodule · v2.4 ensemble",
    student: "Lung Nodule · edge-lite",
    disease: "Lung Nodule",
    compression: "9.1× smaller",
    teacherAuc: 0.968,
    studentAuc: 0.957,
    status: "running",
    progress: 64
  },
  {
    id: "d3",
    teacher: "Cardiovascular Risk · v1.1",
    student: "Cardio · mobile",
    disease: "Cardiovascular Risk",
    compression: "6.8× smaller",
    teacherAuc: 0.952,
    studentAuc: 0,
    status: "queued",
    progress: 0
  }
];

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function DistillationPage() {
  const [teacher, setTeacher] = useState("Breast Cancer · v1.3 ensemble");
  const [target, setTarget] = useState("edge-lite");

  return (
    <div className="distill-page">
      <section className="distill-launch pg-card">
        <div className="pg-card-head">
          <h3>New distillation</h3>
          <span className="pg-hint">teacher → compact student</span>
        </div>
        <div className="pg-field-row">
          <label className="pg-field">
            <span>Teacher model</span>
            <select value={teacher} onChange={(e) => setTeacher(e.target.value)}>
              <option>Breast Cancer · v1.3 ensemble</option>
              <option>Lung Nodule · v2.4 ensemble</option>
              <option>Cardiovascular Risk · v1.1</option>
            </select>
          </label>
          <label className="pg-field">
            <span>Student target</span>
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="edge-lite">Edge-lite (quantized)</option>
              <option value="mobile">Mobile (8-bit)</option>
              <option value="tree">Interpretable tree</option>
            </select>
          </label>
        </div>
        <button type="button" className="primary-button">
          Start distillation
        </button>
      </section>

      <section className="distill-jobs">
        <div className="distill-row distill-row-head">
          <span>Teacher → Student</span>
          <span>Disease</span>
          <span>Compression</span>
          <span>Teacher AUC</span>
          <span>Student AUC</span>
          <span>Status</span>
        </div>
        {JOBS.map((job) => (
          <div key={job.id} className="distill-row">
            <span className="distill-pair">
              <strong>{job.teacher}</strong>
              <span>↳ {job.student}</span>
            </span>
            <span className="distill-disease">{job.disease}</span>
            <span className="distill-compression">{job.compression}</span>
            <span className="model-metric">{pct(job.teacherAuc)}</span>
            <span className="model-metric">
              {job.studentAuc > 0 ? (
                <>
                  {pct(job.studentAuc)}
                  <i className="distill-delta">{((job.studentAuc - job.teacherAuc) * 100).toFixed(1)}</i>
                </>
              ) : (
                "—"
              )}
            </span>
            <span className="distill-status">
              {job.status === "running" ? (
                <span className="distill-progress">
                  <i style={{ width: `${job.progress}%` }} />
                  <em>{job.progress}%</em>
                </span>
              ) : (
                <i className={`status-pill ${job.status === "completed" ? "production" : "draft"}`}>{job.status}</i>
              )}
            </span>
          </div>
        ))}
      </section>
    </div>
  );
}
