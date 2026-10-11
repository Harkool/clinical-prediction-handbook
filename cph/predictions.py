"""Store known-case probabilities explicitly; this object is not a fitted model.

Keras, neural-network and other probability arrays use the same ID contract.
An identity hash detects accidental changes; it is not a digital signature.
"""
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile


FORMAT = "cph-frozen-predictions-v1"
ROLE = "stored probabilities for known case IDs; not a fitted estimator"


def _ids(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("Provide an ordered sequence of case IDs, not a single string")
    identifiers = tuple(values)
    if not identifiers or any(not isinstance(value, str) or not value or
                              value != value.strip() for value in identifiers):
        raise ValueError("Case IDs must be nonempty strings without surrounding whitespace")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case IDs must be unique")
    return identifiers


def _probabilities(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("Provide an ordered sequence of probabilities")
    probabilities = tuple(values)
    if (not probabilities or any(isinstance(value, bool) or not isinstance(value, Real)
                                 for value in probabilities)):
        raise ValueError("Probabilities must be numeric scalars, not labels or nested arrays")
    probabilities = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Probabilities must be finite and between zero and one")
    return probabilities


def metadata_path(path):
    """Pair predictions.csv with predictions.csv.json, preserving the CSV suffix."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def _replace_bytes(path, payload):
    """Replace each file atomically; paired-file consistency is checked on load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="."+path.name+".",suffix=".tmp",
                                     dir=path.parent,delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class FrozenPredictions:
    """Immutable lookup for explicitly identified, already predicted cases.

    ``predict(ids)`` only reorders stored probabilities. It rejects unseen cases
    and cannot infer risk for new features. IDs remain strings to preserve zeros.
    ``source`` describes where the probabilities came from; an optional
    ``model_identity`` records a caller-provided model identity without asserting
    that the model or its clinical validity has been independently verified.
    """
    ids: tuple
    probabilities: tuple
    source: str = "user supplied probabilities"
    model_identity: str = None

    def __post_init__(self):
        identifiers, probabilities = _ids(self.ids), _probabilities(self.probabilities)
        if len(identifiers) != len(probabilities):
            raise ValueError("Every case ID needs exactly one probability")
        if not isinstance(self.source,str) or not self.source.strip():
            raise ValueError("Probability source must be a nonempty description")
        if self.model_identity is not None and (not isinstance(self.model_identity,str)
                                               or not self.model_identity.strip()):
            raise ValueError("Model identity must be a nonempty string or None")
        object.__setattr__(self,"ids",identifiers)
        object.__setattr__(self,"probabilities",probabilities)

    @property
    def prediction_identity(self):
        record = dict(ids=self.ids,probabilities=[value.hex() for value in self.probabilities],
                      source=self.source,model_identity=self.model_identity)
        payload = json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(",",":"))
        return "SHA256:"+hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def predict(self, ids):
        """Return probabilities in the requested ID order; never ignore input."""
        import numpy as np
        identifiers = _ids(ids)
        lookup = dict(zip(self.ids,self.probabilities))
        unknown = [identifier for identifier in identifiers if identifier not in lookup]
        if unknown:
            raise ValueError("No stored prediction for case ID(s): "+", ".join(unknown))
        return np.array([lookup[identifier] for identifier in identifiers],dtype=float)

    def save(self, path):
        """Write CSV and a paired source/identity manifest; returns the CSV path."""
        path = Path(path)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer,lineterminator="\n")
        writer.writerow(("id","probability"))
        writer.writerows((identifier,repr(probability))
                         for identifier,probability in zip(self.ids,self.probabilities))
        payload = buffer.getvalue().encode("utf-8")
        manifest = dict(format=FORMAT,role=ROLE,n=len(self.ids),source=self.source,
                        model_identity=self.model_identity,
                        prediction_identity=self.prediction_identity,
                        csv_sha256=hashlib.sha256(payload).hexdigest())
        metadata = (json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,
                               allow_nan=False)+"\n").encode("utf-8")
        _replace_bytes(path,payload)
        _replace_bytes(metadata_path(path),metadata)
        return path

    @classmethod
    def load(cls, path):
        """Reject altered CSV, metadata, order, IDs, probabilities or provenance."""
        path = Path(path)
        payload = path.read_bytes()
        def invalid_constant(value):
            raise ValueError("Metadata contains nonfinite JSON constant "+value)
        manifest = json.loads(metadata_path(path).read_text(encoding="utf-8"),
                              parse_constant=invalid_constant)
        expected_keys = {"format","role","n","source","model_identity","prediction_identity","csv_sha256"}
        if (not isinstance(manifest,dict) or set(manifest) != expected_keys or manifest.get("format") != FORMAT or
                manifest.get("role") != ROLE or
                manifest.get("csv_sha256") != hashlib.sha256(payload).hexdigest()):
            raise ValueError("Prediction CSV and metadata format/hash do not match")
        stream = io.StringIO(payload.decode("utf-8"),newline="")
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id","probability"]:
            raise ValueError("Prediction CSV must have exactly id,probability columns")
        identifiers, probabilities = [], []
        for row in reader:
            if set(row) != {"id","probability"} or row["id"] is None or row["probability"] is None:
                raise ValueError("Malformed prediction CSV row")
            identifiers.append(row["id"])
            probabilities.append(float(row["probability"]))
        stored = cls(identifiers,probabilities,source=manifest.get("source"),
                     model_identity=manifest.get("model_identity"))
        if (type(manifest.get("n")) is not int or manifest["n"] != len(stored.ids) or
                manifest.get("prediction_identity") != stored.prediction_identity):
            raise ValueError("Prediction identity/row count does not match the paired metadata")
        return stored


def positive_class_probability(model, X):
    """Extract class-1 probabilities from a binary estimator's declared classes.

    Classes must be precisely 0 and 1 in either order. No estimator name or
    neural-network special case is used. For a Keras output or another explicit
    one-dimensional probability vector, construct FrozenPredictions directly.
    """
    import numpy as np
    labels = np.asarray(getattr(model,"classes_",[]))
    if (labels.ndim != 1 or len(labels) != 2 or
            not ((labels[0]==0 and labels[1]==1) or (labels[0]==1 and labels[1]==0))):
        raise ValueError("Estimator classes_ must contain exactly binary labels 0 and 1")
    function = getattr(model,"predict_proba",None)
    if not callable(function):
        raise ValueError("Estimator must implement predict_proba; explicit arrays use FrozenPredictions")
    probabilities = np.asarray(function(X),dtype=float)
    shape = getattr(X,"shape",None)
    try:
        n = shape[0] if shape is not None and len(shape) else len(X)
    except TypeError as error:
        raise ValueError("Prediction input needs a defined number of rows") from error
    if (n < 1 or probabilities.shape != (n,2) or not np.all(np.isfinite(probabilities)) or
            np.any((probabilities < 0) | (probabilities > 1)) or
            not np.allclose(probabilities.sum(axis=1),1.,atol=1e-8,rtol=0)):
        raise ValueError("Estimator predict_proba must return aligned finite two-class probability rows")
    column = int(np.flatnonzero(labels==1)[0])
    return probabilities[:,column].copy()
