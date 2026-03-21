# Threat Model and Security Review Notes

## Threat Model

- Adversary can submit malformed transactions and spam peer endpoints.
- Adversary can send invalid blocks attempting chain poisoning.
- Adversary can exploit non-deterministic policy outputs under high uncertainty.
- Adversary can inspect network traffic and replay unauthenticated requests.

## Defenses in Current Prototype

- Transaction and block schema validation via FastAPI/Pydantic.
- PQC signing and verification integration path for transaction authenticity.
- Safety shield and uncertainty gating in governance action pipeline.
- Audit record and policy hash pipeline for post-hoc accountability.

## Residual Risks

- No production-grade authentication/authorization on RPC endpoints.
- No anti-DoS controls or peer reputation scoring.
- No distributed consensus hardening for byzantine network behavior.
- ZK proof path is placeholder-backed and not on-chain verified.

## Security Review Checklist

- [x] Input validation for transaction and block payloads
- [x] Integration tests for critical RPC endpoints
- [x] Threat assumptions documented
- [ ] Independent cryptographic review
- [ ] Formal protocol verification
- [ ] Adversarial network penetration testing
