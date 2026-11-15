## Change and motivation

Describe the mathematical or software change, affected APIs or equations, and
the issue it addresses. State what a user can now compute or reproduce.

## Mathematical review (check applicable items)

- [ ] The Fourier convention and inverse factor \(1/(2\pi)\) are unchanged,
      or every affected formula and test has been updated.
- [ ] Polynomial zeros, conjugate closure, and multiplicity assumptions are
      documented for changes to the multi-zero family.
- [ ] Signs, factors of \(2\), \(\pi\), and prime-power weights were traced
      from definitions to implementation.
- [ ] \(L^2\) Gram positivity is kept distinct from finite Weil-form
      diagnostics; no finite computation is described as an RH proof.
- [ ] Any finite prime cutoff is labeled, and omitted-tail claims have an
      independently reviewable bound.

## Evidence

- [ ] New or changed formulas have an independent test oracle where practical
      (for example, defining-integral quadrature or symbolic differentiation).
- [ ] Numerical precision, tolerances, convergence checks, and difficult
      parameter cases are stated.
- [ ] The technical note, README, and citation metadata are updated where
      needed.
- [ ] The test suite and paper build pass locally; include commands and
      relevant output below.

~~~text
Commands and results:
~~~

## Review and security

- [ ] I checked user input handling, dependencies, and workflow changes.
- [ ] I reviewed any Codex or other automated review findings and verified
      mathematical suggestions independently. Automated approval is not
      mathematical certification.
