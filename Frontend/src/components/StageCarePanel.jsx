import React from 'react';
import {
  Pill,
  AlertTriangle,
  Utensils,
  CheckCircle2,
  Stethoscope,
  ClipboardCheck,
  CalendarCheck2,
  ShieldAlert,
  BookOpen,
  Clock,
} from 'lucide-react';
import { getStageInformation } from '../utils/stageData';

// Shared list style for structured care-plan bullets.
function BulletList({ items, dotClass = 'bg-slate-500' }) {
  if (!items || items.length === 0) return null;
  return (
    <ul className="space-y-2.5">
      {items.map((item, idx) => (
        <li key={idx} className="flex items-start space-x-2.5 text-xs text-slate-700 leading-relaxed">
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 mt-1.5 ${dotClass}`} />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function SectionCard({ icon: Icon, title, accent, children, footer }) {
  return (
    <div className={`bg-slate-50/70 p-5 rounded-2xl border border-slate-200 hover:border-${accent}-300 transition-colors flex flex-col justify-between`}>
      <div>
        <div className={`flex items-center space-x-2 text-${accent}-700 font-bold text-sm mb-4 pb-2 border-b border-slate-200`}>
          <Icon className={`w-5 h-5 text-${accent}-600`} />
          <span>{title}</span>
        </div>
        {children}
      </div>
      {footer && (
        <div className="mt-4 pt-3 border-t border-slate-200 text-[11px] text-slate-500 font-medium flex items-center justify-between">
          {footer}
        </div>
      )}
    </div>
  );
}

export default function StageCarePanel({ stage, carePlan }) {
  if (!stage) {
    return (
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm text-center">
        <div className="w-12 h-12 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-3">
          <CheckCircle2 className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-bold text-slate-900">No TB Stage Protocol Required</h3>
        <p className="text-sm text-slate-500 max-w-md mx-auto mt-1">
          The X-Ray diagnosis is TB Negative. Standard routine preventive pulmonary wellness guidance applies.
        </p>
      </div>
    );
  }

  // Prefer the server-provided care_plan (authoritative, versioned with the
  // backend's stages.json). Fall back to local stageData if the API didn't
  // return one — e.g. during demo mode without the backend.
  const usingServerPlan = Boolean(carePlan && carePlan.steps);
  const localInfo = usingServerPlan ? null : getStageInformation(stage);
  if (!usingServerPlan && !localInfo) return null;

  const title = usingServerPlan ? carePlan.name : `${localInfo.title} Care Recommendations`;
  const summary = usingServerPlan ? carePlan.summary : localInfo.description;
  const severity = usingServerPlan ? carePlan.severity : null;
  const infectious = usingServerPlan ? carePlan.infectious : null;

  // --- Medication block ---------------------------------------------------
  let medicationItems;
  if (usingServerPlan) {
    const m = carePlan.steps.medication || {};
    medicationItems = [];
    if (m.regimen) medicationItems.push(`Regimen: ${m.regimen}`);
    if (m.dosing_reference) medicationItems.push(`Dosing: ${m.dosing_reference}`);
    (m.alternatives || []).forEach((a) => medicationItems.push(`Alternative: ${a}`));
    (m.adjuncts || []).forEach((a) => medicationItems.push(`Adjunct: ${a}`));
    (m.watch_for || []).forEach((w) => medicationItems.push(`Watch: ${w}`));
    if (m.resistance_check) medicationItems.push(`Resistance: ${m.resistance_check}`);
  } else {
    medicationItems = localInfo.medication || [];
  }

  // --- Precautions / isolation block --------------------------------------
  let precautionItems;
  if (usingServerPlan) {
    precautionItems = [];
    if (carePlan.steps.isolation) precautionItems.push(carePlan.steps.isolation);
    (carePlan.steps.immediate_care || []).forEach((i) => precautionItems.push(i));
  } else {
    precautionItems = localInfo.precautions || [];
  }

  // --- Nutrition block ----------------------------------------------------
  const nutritionItems = usingServerPlan
    ? carePlan.steps.nutrition || []
    : localInfo.diet || [];

  // --- Follow-up block (server only) --------------------------------------
  const followUp = usingServerPlan ? carePlan.steps.follow_up || {} : null;
  const followUpItems = followUp
    ? [
        followUp.visits && `Visits — ${followUp.visits}`,
        followUp.labs && `Labs — ${followUp.labs}`,
        followUp.sputum && `Sputum — ${followUp.sputum}`,
        followUp.imaging && `Imaging — ${followUp.imaging}`,
        followUp.contact_tracing && `Contact tracing — ${followUp.contact_tracing}`,
        followUp.rehab && `Rehab — ${followUp.rehab}`,
        followUp.endpoint && `Endpoint — ${followUp.endpoint}`,
      ].filter(Boolean)
    : [];

  const redFlags = usingServerPlan ? carePlan.red_flags || [] : [];
  const education = usingServerPlan ? carePlan.patient_education || [] : [];

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-2">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg border border-indigo-100">
            S{stage}
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900 tracking-tight">{title}</h3>
            <p className="text-xs text-slate-500">
              Structured care pathway: immediate steps, medication, isolation, nutrition, follow-up.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 self-start sm:self-auto">
          {severity && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              Severity: {severity}
            </span>
          )}
          {infectious === true && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200">
              Infectious
            </span>
          )}
          {infectious === false && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              Non-infectious
            </span>
          )}
        </div>
      </div>

      {summary && (
        <p className="text-sm text-slate-600 italic bg-slate-50 p-3.5 rounded-xl border border-slate-200">
          "{summary}"
        </p>
      )}

      {/* Primary care categories */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <SectionCard
          icon={Pill}
          title="Medication"
          accent="sky"
          footer={<><span>Clinical Action</span><span className="text-sky-700">Verify dose by weight &amp; DST</span></>}
        >
          <BulletList items={medicationItems} dotClass="bg-sky-500" />
        </SectionCard>

        <SectionCard
          icon={ShieldAlert}
          title="Isolation &amp; Immediate Care"
          accent="amber"
          footer={<><span>Infection Control</span><span className="text-amber-700">Strict compliance</span></>}
        >
          <BulletList items={precautionItems} dotClass="bg-amber-500" />
        </SectionCard>

        <SectionCard
          icon={Utensils}
          title="Nutrition Support"
          accent="emerald"
          footer={<><span>Nutritional Status</span><span className="text-emerald-700">High-protein, micronutrients</span></>}
        >
          <BulletList items={nutritionItems} dotClass="bg-emerald-500" />
        </SectionCard>
      </div>

      {/* Follow-up schedule + red flags + education — server-plan only */}
      {usingServerPlan && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <SectionCard
            icon={CalendarCheck2}
            title="Follow-up Schedule"
            accent="violet"
            footer={<><span>Longitudinal plan</span><span className="text-violet-700">DOT &amp; monitoring</span></>}
          >
            <BulletList items={followUpItems} dotClass="bg-violet-500" />
          </SectionCard>

          <SectionCard
            icon={AlertTriangle}
            title="Red-Flag Symptoms"
            accent="rose"
            footer={<><span>Escalate immediately</span><span className="text-rose-700">ER / specialist</span></>}
          >
            <BulletList items={redFlags} dotClass="bg-rose-500" />
          </SectionCard>

          <SectionCard
            icon={BookOpen}
            title="Patient Education"
            accent="teal"
            footer={<><span>Shared with patient</span><span className="text-teal-700">Printed handout</span></>}
          >
            <BulletList items={education} dotClass="bg-teal-500" />
          </SectionCard>
        </div>
      )}

      {/* Disclaimer */}
      {usingServerPlan && carePlan.disclaimer && (
        <div className="flex items-start space-x-2 text-[11px] text-slate-500 bg-slate-50 border border-slate-200 rounded-xl p-3">
          <Stethoscope className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>{carePlan.disclaimer}</span>
        </div>
      )}
    </div>
  );
}
