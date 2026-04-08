from __future__ import annotations

import argparse
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


ROOT_CA_NAME = "Mobile Alert Local Dev Root"


def default_dns_names() -> list[str]:
    names = {"localhost", socket.gethostname()}
    fqdn = socket.getfqdn()
    if fqdn:
        names.add(fqdn)
    return sorted(name for name in names if name)


def default_ip_addresses() -> list[str]:
    candidates = {"127.0.0.1"}
    hostnames = {socket.gethostname(), socket.getfqdn(), "localhost"}

    for hostname in hostnames:
        if not hostname:
            continue
        try:
            for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, socket.AF_INET):
                if family == socket.AF_INET and sockaddr:
                    ip = sockaddr[0]
                    if not ip.startswith("169.254."):
                        candidates.add(ip)
        except OSError:
            continue

    return sorted(candidates)


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def write_pem(path: Path, blocks: list[bytes]) -> None:
    path.write_bytes(b"".join(blocks))


def build_root_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ROOT_CA_NAME)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365 * 5))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    return key, cert


def build_server_cert(
    root_key: rsa.RSAPrivateKey,
    root_cert: x509.Certificate,
    dns_names: list[str],
    ip_addresses: list[str],
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dns_names[0])])
    san_entries: list[x509.GeneralName] = [x509.DNSName(name) for name in dns_names]
    san_entries.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in ip_addresses)

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(root_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365 * 2))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(private_key=root_key, algorithm=hashes.SHA256())
    )
    return key, cert


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local HTTPS certificate for the mobile alert server.")
    parser.add_argument(
        "--certs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "certs",
    )
    parser.add_argument("--dns-name", action="append", default=[])
    parser.add_argument("--ip-address", action="append", default=[])
    args = parser.parse_args()

    dns_names = unique(args.dns_name + default_dns_names())
    ip_addresses = unique(args.ip_address + default_ip_addresses())

    certs_dir = args.certs_dir.resolve()
    certs_dir.mkdir(parents=True, exist_ok=True)

    root_key, root_cert = build_root_ca()
    server_key, server_cert = build_server_cert(root_key, root_cert, dns_names, ip_addresses)

    root_ca_pem = root_cert.public_bytes(serialization.Encoding.PEM)
    root_ca_der = root_cert.public_bytes(serialization.Encoding.DER)
    server_cert_pem = server_cert.public_bytes(serialization.Encoding.PEM)
    server_key_pem = server_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    write_pem(certs_dir / "mobile-alert-root-ca.pem", [root_ca_pem])
    (certs_dir / "mobile-alert-root-ca.cer").write_bytes(root_ca_der)
    write_pem(certs_dir / "mobile-alert-local-cert.pem", [server_cert_pem, root_ca_pem])
    write_pem(certs_dir / "mobile-alert-local-key.pem", [server_key_pem])

    print("Generated local HTTPS certificate files:")
    print(f"  Root CA (.cer): {certs_dir / 'mobile-alert-root-ca.cer'}")
    print(f"  Root CA (.pem): {certs_dir / 'mobile-alert-root-ca.pem'}")
    print(f"  Server cert:    {certs_dir / 'mobile-alert-local-cert.pem'}")
    print(f"  Server key:     {certs_dir / 'mobile-alert-local-key.pem'}")
    print()
    print(f"SAN hostnames: {', '.join(dns_names)}")
    print(f"SAN IPs:       {', '.join(ip_addresses)}")
    print()
    print("Install the root CA on the Android phone, then open https://<your-pc-ip>:8766/phone/")


if __name__ == "__main__":
    main()
